/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.elasticsearch.xpack.ml.datafeed;

import org.apache.logging.log4j.Level;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.DocWriteResponse;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.cluster.metadata.ProjectMetadata;
import org.elasticsearch.common.settings.SecureString;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.ThreadContext;
import org.elasticsearch.core.Tuple;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.test.MockLog;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xpack.core.ml.action.DeleteDatafeedAction;
import org.elasticsearch.xpack.core.ml.action.PutDatafeedAction;
import org.elasticsearch.xpack.core.ml.action.UpdateDatafeedAction;
import org.elasticsearch.xpack.core.ml.datafeed.DatafeedConfig;
import org.elasticsearch.xpack.core.ml.datafeed.DatafeedUpdate;
import org.elasticsearch.xpack.core.rollup.action.GetRollupIndexCapsAction;
import org.elasticsearch.xpack.core.security.SecurityContext;
import org.elasticsearch.xpack.core.security.action.user.HasPrivilegesAction;
import org.elasticsearch.xpack.core.security.action.user.HasPrivilegesResponse;
import org.elasticsearch.xpack.core.security.authc.Authentication;
import org.elasticsearch.xpack.core.security.authc.AuthenticationTestHelper;
import org.elasticsearch.xpack.core.security.authc.support.SecondaryAuthentication;
import org.elasticsearch.xpack.core.security.cloud.CloudCredential;
import org.elasticsearch.xpack.core.security.cloud.CloudCredentialManager;
import org.elasticsearch.xpack.core.security.cloud.InternalCloudApiKeyService;
import org.elasticsearch.xpack.core.security.cloud.PersistedCloudCredential;
import org.elasticsearch.xpack.core.security.user.User;
import org.elasticsearch.xpack.ml.MachineLearningExtension;
import org.elasticsearch.xpack.ml.datafeed.persistence.DatafeedConfigProvider;
import org.elasticsearch.xpack.ml.job.persistence.JobConfigProvider;
import org.mockito.Mockito;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.notNullValue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DatafeedManagerTests extends ESTestCase {

    private static MachineLearningExtension mockMlExtension(
        CloudCredentialManager credentialManager,
        InternalCloudApiKeyService apiKeyService
    ) {
        MachineLearningExtension ext = mock(MachineLearningExtension.class);
        when(ext.getCloudCredentialManager()).thenReturn(credentialManager);
        when(ext.getCloudApiKeyService()).thenReturn(apiKeyService);
        return ext;
    }

    private static void mockGrantSucceeds(InternalCloudApiKeyService apiKeyService, PersistedCloudCredential persisted) {
        Authentication authentication = AuthenticationTestHelper.builder().build();
        doAnswer(invocation -> {
            @SuppressWarnings("unchecked")
            ActionListener<InternalCloudApiKeyService.CloudGrantApiKeyResult> listener = (ActionListener<
                InternalCloudApiKeyService.CloudGrantApiKeyResult>) invocation.getArguments()[2];
            listener.onResponse(new InternalCloudApiKeyService.CloudGrantApiKeyResult(persisted, authentication));
            return null;
        }).when(apiKeyService).grantCloudAuthentication(any(CloudCredential.class), anyString(), any());
    }

    @SuppressWarnings("unchecked")
    public void testDeleteDatafeed_GivenPersistedCloudCredential_LogsSkippingRevocationBeforeDeletionChain() {
        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        CloudCredentialManager credentialManager = mock(CloudCredentialManager.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(credentialManager, apiKeyService);
        Client client = mock(Client.class);

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            mock(JobConfigProvider.class),
            NamedXContentRegistry.EMPTY,
            Settings.EMPTY,
            client,
            mlExtension
        );

        PersistedCloudCredential cred = new PersistedCloudCredential("test-key-id", new SecureString("secret".toCharArray()));
        DatafeedConfig.Builder builder = new DatafeedConfig.Builder("test-datafeed", "test-job");
        builder.setIndices(List.of("logs-*"));
        builder.setCloudInternalCredential(cred);
        DatafeedConfig datafeedConfig = builder.build();

        doAnswer(invocation -> {
            ActionListener<DatafeedConfig.Builder> listener = (ActionListener<DatafeedConfig.Builder>) invocation.getArguments()[2];
            listener.onResponse(new DatafeedConfig.Builder(datafeedConfig));
            return null;
        }).when(datafeedConfigProvider).getDatafeedConfig(eq("test-datafeed"), isNull(), any());

        ClusterState clusterState = mockClusterStateWithNoTasks();
        DeleteDatafeedAction.Request request = new DeleteDatafeedAction.Request("test-datafeed");

        MockLog.assertThatLogger(
            () -> manager.deleteDatafeed(request, clusterState, ActionListener.wrap(r -> fail("expected incomplete chain"), e -> {})),
            DatafeedManager.class,
            new MockLog.PatternSeenEventExpectation(
                "skip revoke on delete",
                DatafeedManager.class.getCanonicalName(),
                Level.WARN,
                ".*Skipping revocation of cloud API key \\[test-key-id\\].*on deletion.*"
            )
        );

        verify(datafeedConfigProvider).getDatafeedConfig(eq("test-datafeed"), isNull(), any());
        verify(apiKeyService, never()).grantCloudAuthentication(any(), anyString(), any());
    }

    @SuppressWarnings("unchecked")
    public void testDeleteDatafeed_GivenNoPersistedCloudCredential_DoesNotMintOrRevokeKeys() {
        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(mock(CloudCredentialManager.class), apiKeyService);
        Client client = mock(Client.class);

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            mock(JobConfigProvider.class),
            NamedXContentRegistry.EMPTY,
            Settings.EMPTY,
            client,
            mlExtension
        );

        DatafeedConfig.Builder builder = new DatafeedConfig.Builder("test-datafeed", "test-job");
        builder.setIndices(List.of("logs-*"));
        DatafeedConfig datafeedConfig = builder.build();

        doAnswer(invocation -> {
            ActionListener<DatafeedConfig.Builder> listener = (ActionListener<DatafeedConfig.Builder>) invocation.getArguments()[2];
            listener.onResponse(new DatafeedConfig.Builder(datafeedConfig));
            return null;
        }).when(datafeedConfigProvider).getDatafeedConfig(eq("test-datafeed"), isNull(), any());

        ClusterState clusterState = mockClusterStateWithNoTasks();
        DeleteDatafeedAction.Request request = new DeleteDatafeedAction.Request("test-datafeed");

        manager.deleteDatafeed(request, clusterState, ActionListener.wrap(r -> fail("expected incomplete chain"), e -> {}));

        verify(datafeedConfigProvider).getDatafeedConfig(eq("test-datafeed"), isNull(), any());
        verify(apiKeyService, never()).grantCloudAuthentication(any(), anyString(), any());
    }

    private static ClusterState mockClusterStateWithNoTasks() {
        ClusterState clusterState = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        ProjectMetadata projectMetadata = mock(ProjectMetadata.class);
        when(clusterState.getMetadata()).thenReturn(metadata);
        when(clusterState.metadata()).thenReturn(metadata);
        when(metadata.getProject()).thenReturn(projectMetadata);
        when(projectMetadata.custom(any())).thenReturn(null);
        when(projectMetadata.getIndicesLookup()).thenReturn(Collections.emptySortedMap());
        return clusterState;
    }

    @SuppressWarnings("unchecked")
    private static void stubClientForSecurityPutPath(Client client, ThreadPool threadPool) {
        when(client.threadPool()).thenReturn(threadPool);
        doAnswer(invocation -> {
            ActionListener<GetRollupIndexCapsAction.Response> listener = invocation.getArgument(2);
            listener.onResponse(new GetRollupIndexCapsAction.Response());
            return null;
        }).when(client).execute(same(GetRollupIndexCapsAction.INSTANCE), any(), any());
        doAnswer(invocation -> {
            ActionListener<HasPrivilegesResponse> listener = invocation.getArgument(2);
            listener.onResponse(new HasPrivilegesResponse());
            return null;
        }).when(client).execute(same(HasPrivilegesAction.INSTANCE), any(), any());
    }

    private static SecurityContext mockSecurityContextWithUser(String principal) {
        SecurityContext securityContext = mock(SecurityContext.class);
        User user = mock(User.class);
        when(user.principal()).thenReturn(principal);
        when(securityContext.getUser()).thenReturn(user);
        return securityContext;
    }

    @SuppressWarnings("unchecked")
    public void testPutDatafeedWithSecurityAndCpsCredentialShouldGrantAndPersistKey() {
        Settings settings = Settings.builder().put("serverless.cross_project.enabled", true).put("xpack.security.enabled", true).build();

        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        CloudCredentialManager credentialManager = mock(CloudCredentialManager.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(credentialManager, apiKeyService);
        JobConfigProvider jobConfigProvider = mock(JobConfigProvider.class);
        Client client = mock(Client.class);
        ThreadPool threadPool = mock(ThreadPool.class);
        ThreadContext threadContext = new ThreadContext(Settings.EMPTY);
        when(threadPool.getThreadContext()).thenReturn(threadContext);

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            jobConfigProvider,
            NamedXContentRegistry.EMPTY,
            settings,
            client,
            mlExtension
        );

        when(credentialManager.hasCloudManagedCredential(any())).thenReturn(true);
        CloudCredential extractedCredential = new CloudCredential(new SecureString("caller-token".toCharArray()));
        when(credentialManager.extractCloudManagedCredential(any())).thenReturn(extractedCredential);

        PersistedCloudCredential persisted = new PersistedCloudCredential("minted-key-id", new SecureString("secret".toCharArray()));
        mockGrantSucceeds(apiKeyService, persisted);

        stubClientForSecurityPutPath(client, threadPool);

        doAnswer(invocation -> {
            ActionListener<Set<String>> listener = (ActionListener<Set<String>>) invocation.getArguments()[1];
            listener.onResponse(Collections.emptySet());
            return null;
        }).when(datafeedConfigProvider).findDatafeedIdsForJobIds(any(), any());

        doAnswer(invocation -> {
            ActionListener<Boolean> listener = (ActionListener<Boolean>) invocation.getArguments()[1];
            listener.onResponse(Boolean.TRUE);
            return null;
        }).when(jobConfigProvider).validateDatafeedJob(any(), any());

        doAnswer(invocation -> {
            ActionListener<Tuple<DatafeedConfig, DocWriteResponse>> listener = (ActionListener<
                Tuple<DatafeedConfig, DocWriteResponse>>) invocation.getArguments()[2];
            DatafeedConfig cfg = invocation.getArgument(0);
            listener.onResponse(Tuple.tuple(cfg, mock(DocWriteResponse.class)));
            return null;
        }).when(datafeedConfigProvider).putDatafeedConfig(any(), any(), any());

        DatafeedConfig.Builder builder = new DatafeedConfig.Builder("test-datafeed", "test-job");
        builder.setIndices(List.of("logs-*"));
        PutDatafeedAction.Request request = new PutDatafeedAction.Request(builder.build());

        SecurityContext securityContext = mockSecurityContextWithUser("df-user");

        AtomicReference<PutDatafeedAction.Response> response = new AtomicReference<>();
        manager.putDatafeed(
            request,
            mockClusterStateForUpdate(),
            securityContext,
            threadPool,
            ActionListener.wrap(response::set, e -> fail("unexpected failure: " + e))
        );

        assertThat(response.get(), notNullValue());
        assertThat(response.get().getResponse().getCloudInternalCredential(), equalTo(persisted));
        verify(apiKeyService).grantCloudAuthentication(eq(extractedCredential), eq("datafeed:test-datafeed"), any());
    }

    @SuppressWarnings("unchecked")
    public void testPutDatafeedWithSecurityAndSecondaryAuthShouldGrantUnderSecondaryPrincipal() {
        Settings settings = Settings.builder().put("serverless.cross_project.enabled", true).put("xpack.security.enabled", true).build();

        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        CloudCredentialManager credentialManager = mock(CloudCredentialManager.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(credentialManager, apiKeyService);
        JobConfigProvider jobConfigProvider = mock(JobConfigProvider.class);
        Client client = mock(Client.class);
        ThreadPool threadPool = mock(ThreadPool.class);
        ThreadContext threadContext = new ThreadContext(Settings.EMPTY);
        when(threadPool.getThreadContext()).thenReturn(threadContext);

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            jobConfigProvider,
            NamedXContentRegistry.EMPTY,
            settings,
            client,
            mlExtension
        );

        when(credentialManager.hasCloudManagedCredential(any())).thenReturn(true);
        CloudCredential extractedCredential = new CloudCredential(new SecureString("caller-token".toCharArray()));
        when(credentialManager.extractCloudManagedCredential(any())).thenReturn(extractedCredential);

        PersistedCloudCredential persisted = new PersistedCloudCredential("minted-key-id-2", new SecureString("secret".toCharArray()));
        mockGrantSucceeds(apiKeyService, persisted);

        stubClientForSecurityPutPath(client, threadPool);

        doAnswer(invocation -> {
            ActionListener<Set<String>> listener = (ActionListener<Set<String>>) invocation.getArguments()[1];
            listener.onResponse(Collections.emptySet());
            return null;
        }).when(datafeedConfigProvider).findDatafeedIdsForJobIds(any(), any());

        doAnswer(invocation -> {
            ActionListener<Boolean> listener = (ActionListener<Boolean>) invocation.getArguments()[1];
            listener.onResponse(Boolean.TRUE);
            return null;
        }).when(jobConfigProvider).validateDatafeedJob(any(), any());

        doAnswer(invocation -> {
            ActionListener<Tuple<DatafeedConfig, DocWriteResponse>> listener = (ActionListener<
                Tuple<DatafeedConfig, DocWriteResponse>>) invocation.getArguments()[2];
            DatafeedConfig cfg = invocation.getArgument(0);
            listener.onResponse(Tuple.tuple(cfg, mock(DocWriteResponse.class)));
            return null;
        }).when(datafeedConfigProvider).putDatafeedConfig(any(), any(), any());

        DatafeedConfig.Builder builder = new DatafeedConfig.Builder("test-datafeed", "test-job");
        builder.setIndices(List.of("logs-*"));
        PutDatafeedAction.Request request = new PutDatafeedAction.Request(builder.build());

        SecurityContext securityContext = mockSecurityContextWithUser("df-user");
        SecondaryAuthentication secondaryAuth = mock(SecondaryAuthentication.class);
        when(secondaryAuth.wrap(any(Runnable.class))).thenAnswer(invocation -> {
            Runnable inner = invocation.getArgument(0);
            return (Runnable) inner::run;
        });
        when(securityContext.getSecondaryAuthentication()).thenReturn(secondaryAuth);

        AtomicReference<PutDatafeedAction.Response> response = new AtomicReference<>();
        manager.putDatafeed(
            request,
            mockClusterStateForUpdate(),
            securityContext,
            threadPool,
            ActionListener.wrap(response::set, e -> fail("unexpected failure: " + e))
        );

        assertThat(response.get().getResponse().getCloudInternalCredential(), equalTo(persisted));
        Mockito.verify(secondaryAuth, Mockito.atLeast(2)).wrap(any(Runnable.class));
    }

    /**
     * If downstream work fails after a cloud API key was granted, the failure is still propagated (revocation is deferred until
     * {@code InternalCloudApiKeyService} exposes revoke).
     */
    @SuppressWarnings("unchecked")
    public void testPutDatafeed_PropagatesFailureWhenDownstreamFailsAfterGrant() {
        Settings settings = Settings.builder().put("serverless.cross_project.enabled", true).put("xpack.security.enabled", false).build();

        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        CloudCredentialManager credentialManager = mock(CloudCredentialManager.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(credentialManager, apiKeyService);
        JobConfigProvider jobConfigProvider = mock(JobConfigProvider.class);
        Client client = mock(Client.class);
        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.getThreadContext()).thenReturn(new ThreadContext(Settings.EMPTY));

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            jobConfigProvider,
            NamedXContentRegistry.EMPTY,
            settings,
            client,
            mlExtension
        );

        when(credentialManager.hasCloudManagedCredential(any())).thenReturn(true);
        CloudCredential extractedCredential = new CloudCredential(new SecureString("test-token".toCharArray()));
        when(credentialManager.extractCloudManagedCredential(any())).thenReturn(extractedCredential);

        PersistedCloudCredential persisted = new PersistedCloudCredential("new-key-id", new SecureString("secret".toCharArray()));
        mockGrantSucceeds(apiKeyService, persisted);

        doAnswer(invocation -> {
            ActionListener<Set<String>> listener = (ActionListener<Set<String>>) invocation.getArguments()[1];
            listener.onFailure(new RuntimeException("simulated downstream failure"));
            return null;
        }).when(datafeedConfigProvider).findDatafeedIdsForJobIds(any(), any());

        DatafeedConfig.Builder builder = new DatafeedConfig.Builder("test-datafeed", "test-job");
        builder.setIndices(List.of("logs-*"));
        PutDatafeedAction.Request request = new PutDatafeedAction.Request(builder.build());

        AtomicReference<Exception> failure = new AtomicReference<>();
        manager.putDatafeed(
            request,
            mockClusterStateWithNoTasks(),
            null,
            threadPool,
            ActionListener.wrap(r -> fail("Expected failure"), failure::set)
        );

        assertThat(failure.get(), notNullValue());
        assertThat(failure.get().getMessage(), containsString("simulated downstream failure"));
    }

    /**
     * CPS update path uses {@link DatafeedConfigProvider#updateDatefeedConfig} with a persisted credential; if that update fails,
     * the error is propagated to the listener.
     */
    @SuppressWarnings("unchecked")
    public void testUpdateDatafeed_PropagatesFailureWhenUpdateConfigFails() {
        Settings settings = Settings.builder().put("serverless.cross_project.enabled", true).put("xpack.security.enabled", false).build();

        DatafeedConfigProvider datafeedConfigProvider = mock(DatafeedConfigProvider.class);
        CloudCredentialManager credentialManager = mock(CloudCredentialManager.class);
        InternalCloudApiKeyService apiKeyService = mock(InternalCloudApiKeyService.class);
        MachineLearningExtension mlExtension = mockMlExtension(credentialManager, apiKeyService);
        JobConfigProvider jobConfigProvider = mock(JobConfigProvider.class);
        Client client = mock(Client.class);
        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.getThreadContext()).thenReturn(new ThreadContext(Settings.EMPTY));

        DatafeedManager manager = new DatafeedManager(
            datafeedConfigProvider,
            jobConfigProvider,
            NamedXContentRegistry.EMPTY,
            settings,
            client,
            mlExtension
        );

        when(credentialManager.hasCloudManagedCredential(any())).thenReturn(true);
        CloudCredential extractedCredential = new CloudCredential(new SecureString("test-token".toCharArray()));
        when(credentialManager.extractCloudManagedCredential(any())).thenReturn(extractedCredential);

        PersistedCloudCredential persisted = new PersistedCloudCredential("update-key-id", new SecureString("secret".toCharArray()));
        mockGrantSucceeds(apiKeyService, persisted);

        doAnswer(invocation -> {
            ActionListener<Tuple<DatafeedConfig, PersistedCloudCredential>> listener = (ActionListener<
                Tuple<DatafeedConfig, PersistedCloudCredential>>) invocation.getArguments()[5];
            listener.onFailure(new RuntimeException("simulated update failure"));
            return null;
        }).when(datafeedConfigProvider)
            .updateDatefeedConfig(
                anyString(),
                any(DatafeedUpdate.class),
                any(Map.class),
                any(PersistedCloudCredential.class),
                any(),
                any()
            );

        DatafeedUpdate.Builder updateBuilder = new DatafeedUpdate.Builder("test-datafeed");
        updateBuilder.setIndices(List.of("new-logs-*"));
        UpdateDatafeedAction.Request request = new UpdateDatafeedAction.Request(updateBuilder.build());

        ClusterState clusterState = mockClusterStateForUpdate();

        AtomicReference<Exception> failure = new AtomicReference<>();
        manager.updateDatafeed(request, clusterState, null, threadPool, ActionListener.wrap(r -> fail("Expected failure"), failure::set));

        assertThat(failure.get(), notNullValue());
        assertThat(failure.get().getMessage(), containsString("simulated update failure"));
    }

    /**
     * Creates a mock ClusterState suitable for both task checks and ElasticsearchMappings.addDocMappingIfMissing.
     */
    private static ClusterState mockClusterStateForUpdate() {
        ClusterState clusterState = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        ProjectMetadata projectMetadata = mock(ProjectMetadata.class);
        when(clusterState.getMetadata()).thenReturn(metadata);
        when(clusterState.metadata()).thenReturn(metadata);
        when(metadata.getProject()).thenReturn(projectMetadata);
        when(projectMetadata.custom(any())).thenReturn(null);
        when(projectMetadata.getIndicesLookup()).thenReturn(Collections.emptySortedMap());
        return clusterState;
    }
}

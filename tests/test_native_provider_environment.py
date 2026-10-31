"""Provider credentials needed by nested Agent runtimes cross the SDK boundary."""

from contextlib import contextmanager
import json
import shutil
from types import SimpleNamespace

from agentbench.runtime.agentcontainer.config import AgentContainerConfig
from agentbench.runtime.contracts import EnvironmentSecretResolver
from agentbench.sdk.plugin.native.plugin import plugin
from agentbench.sdk.plugin.native.runner import NativeRunner


def test_native_runner_forwards_selected_provider_credential_to_controller(tmp_path, monkeypatch):
    root = tmp_path / 'unit'
    root.mkdir()
    (root / 'agent.toml').write_text(
        'schema_version = "defuzex-bench.agent.v2"\n'
        'agent_id = "provider-fixture"\nframework = "acp"\n'
        '[runtime]\ntype = "docker"\nenv_keys = []\nsecret_env_keys = []\n'
        '[build]\ncontext = "."\ndockerfile = "Dockerfile"\n'
        '[launch]\nargv = ["python", "agent.py"]\n'
        '[llm_interception]\nmode = "observe"\n')
    (root / 'Dockerfile').write_text('FROM python:3.12-slim\nUSER 10001\n')
    provider_catalog = tmp_path / 'providers.toml'
    provider_catalog.write_text(
        'priority = ["stepfun"]\n'
        '[providers.stepfun]\n'
        'credential_env = "STEPFUN_API_KEY"\n'
        'model_env = "STEPFUN_MODEL"\n'
        'base_url_env = "STEPFUN_BASE_URL"\n'
        'base_url = "https://api.stepfun.com/step_plan/v1"\n'
        'target_plugin = "compatible-json"\n'
        '[providers.stepfun.endpoint_paths]\n'
        '"/chat/completions" = "/chat/completions"\n')
    secret = 'synthetic-provider-token-for-forwarding'
    environ = {
        'ABB_MODEL': 'step-5-preview',
        'ABB_MODEL_PROVIDERS_CONFIG': str(provider_catalog),
        'ABB_MODEL_PROVIDER': 'stepfun',
        'STEPFUN_API_KEY': secret,
    }
    context = SimpleNamespace(environ=environ, model='step-5-preview')
    agent = SimpleNamespace(path=root, agent_id='provider-fixture', framework='acp')
    staged_root = tmp_path / 'staged'

    @contextmanager
    def stage_agent(source, **kwargs):
        del kwargs
        staged_root.mkdir()
        shutil.copy2(source.path / 'agent.toml', staged_root / 'agent.toml')
        shutil.copy2(source.path / 'Dockerfile', staged_root / 'Dockerfile')
        yield SimpleNamespace(path=staged_root, agent_id=source.agent_id,
                              framework=source.framework)

    monkeypatch.setattr(plugin, 'stage_agent', stage_agent)
    runner = NativeRunner(plugin, context=context, options={})
    with runner._stage_agent(context)(agent) as staged:
        manifest_path = staged.path / 'agent.toml'
        from agentbench.runtime.agentcontainer.config import tomllib
        manifest = tomllib.loads(manifest_path.read_text())
        assert manifest['runtime']['worker_env_keys'] == ['STEPFUN_API_KEY']
        assert secret not in manifest_path.read_text()

        runtime = AgentContainerConfig.from_agent_dir(
            staged.path, secret_resolver=EnvironmentSecretResolver(environ),
            environ=environ)
        assert runtime.environment['STEPFUN_API_KEY'] == secret
        assert runtime.environment['ABB_AGENT_MODEL'] == 'step-5-preview'

    original = (root / 'agent.toml').read_text()
    assert 'worker_env_keys' not in original

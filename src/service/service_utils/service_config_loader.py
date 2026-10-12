import os
import re

from dynaconf import Dynaconf
from loguru import logger

from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from engine_utils.directory_info import DirectoryInfo
from service.service_data_models.logger_config_data import LoggerConfigData
from service.service_data_models.service_config_data import ServiceConfigData

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env_vars(value):
    """Replaces ${VAR} in config strings with the environment value; unset variables are left as-is."""
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                logger.warning(f"Config references environment variable {name}, which is not set")
                return match.group(0)
            return os.environ[name]
        return _ENV_VAR_RE.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def load_configs(in_args):
    os.environ["ENV_FOR_DYNACONF"] = in_args.env
    base_dir = DirectoryInfo.get_project_dir()
    if os.path.isabs(in_args.config):
        config_path = in_args.config
    else:
        config_path = os.path.join(base_dir, in_args.config)
    if not os.path.isfile(config_path):
        logger.error(f"Config file {config_path} not found!")
        exit(1)

    logger.info(f"Load config with env {in_args.env} from {config_path}")
    config = Dynaconf(
        settings_files=[config_path],
        environments=True,
        load_dotenv=True
    )

    out_logger_config = LoggerConfigData.model_validate(config.get("logger", {}))
    out_service_config = ServiceConfigData.model_validate(config.get("service", {}))
    # Dynaconf uppercases all keys - normalize to lowercase for pydantic model_validate
    raw_engine = config.get("chat_engine", {})
    if hasattr(raw_engine, 'to_dict'):
        raw_engine = raw_engine.to_dict()
    elif hasattr(raw_engine, 'as_dict'):
        raw_engine = raw_engine.as_dict()
    if isinstance(raw_engine, dict):
        raw_engine = {k.lower(): v for k, v in raw_engine.items()}
    raw_engine = expand_env_vars(raw_engine)
    out_engine_config = ChatEngineConfigModel.model_validate(raw_engine)
    return out_logger_config, out_service_config, out_engine_config


from typing import Type, List

from pydantic import BaseModel

from service.rtc_service.base_turn_provider import (
    BaseRtcTurnProvider,
    BaseRtcTurnEntity,
)


class StunServerConfigData(BaseModel):
    stun_urls: List[str]


class StunServerProvider(BaseRtcTurnProvider):
    """STUN-only provider (no credentials required)."""

    def get_config_model(self) -> Type[BaseModel]:
        return StunServerConfigData

    def prepare_rtc_configuration(self, config: BaseModel) -> BaseRtcTurnEntity:
        result = BaseRtcTurnEntity()
        result.rtc_configuration = {
            "iceServers": [{"urls": url} for url in config.stun_urls]
        }
        return result

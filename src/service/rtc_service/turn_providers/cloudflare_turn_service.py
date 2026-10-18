import json
import urllib.error
import urllib.request
from typing import Type

from loguru import logger
from pydantic import BaseModel, Field

from service.rtc_service.base_turn_provider import (
    BaseRtcTurnProvider,
    BaseRtcTurnEntity,
)


class CloudflareTurnConfigData(BaseModel):
    key_id: str
    api_token: str
    ttl: int = Field(default=86400)


class CloudflareTurnProvider(BaseRtcTurnProvider):

    def get_config_model(self) -> Type[BaseModel]:
        return CloudflareTurnConfigData

    def prepare_rtc_configuration(self, config: BaseModel):
        config = CloudflareTurnConfigData.model_validate(config)
        url = (
            "https://rtc.live.cloudflare.com/v1/turn/keys/"
            f"{config.key_id}/credentials/generate-ice-servers"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps({"ttl": config.ttl}).encode(),
            headers={
                "Authorization": f"Bearer {config.api_token}",
                "Content-Type": "application/json",
                # Cloudflare rejects the default Python user agent with error 1010.
                "User-Agent": "open-avatar-chat",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            logger.error(f"Cloudflare TURN credential request failed: {exc.code} {detail}")
            raise
        ice_servers = []
        for server in payload.get("iceServers", []):
            urls = server.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            urls = [item for item in urls if ":53" not in item]
            if urls:
                cleaned = dict(server)
                cleaned["urls"] = urls
                ice_servers.append(cleaned)
        result = BaseRtcTurnEntity()
        result.rtc_configuration = {
            "iceServers": ice_servers,
            # The phone is off the LAN, so media has to go through the relay.
            "iceTransportPolicy": "relay",
        }
        return result

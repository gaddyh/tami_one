import httpx


class GreenApiClient:
    def __init__(self, *, id_instance: str, api_token: str, base_url: str):
        self.id_instance = id_instance
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    def _url(self, method: str) -> str:
        return (
            f"{self.base_url}/waInstance{self.id_instance}"
            f"/{method}/{self.api_token}"
        )

    async def get_chats(self, count: int = 50) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(self._url("getChats"), params={"count": count})
            r.raise_for_status()
            return r.json()

    async def get_group_data(self, group_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                self._url("getGroupData"),
                json={"groupId": group_id},
            )
            r.raise_for_status()
            return r.json()

    async def get_chat_history(self, chat_id: str, count: int = 30) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                self._url("getChatHistory"),
                json={"chatId": chat_id, "count": count},
            )
            r.raise_for_status()
            return r.json()

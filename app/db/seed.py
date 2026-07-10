"""Seed the database with demo data."""

from __future__ import annotations

from sqlmodel import Session

from app.db.cache import load_cache
from app.db.engine import engine, init_db
from app.db.models import Tenant, TenantKind, WhatsAppAccount, WhatsAppProvider


def run_seed(name: str , instance_id: str, chat_id: str, overwrite: bool = False) -> dict[str, str]:
    init_db(overwrite=overwrite)

    with Session(engine) as session:
        tenant = Tenant(name=name, kind=TenantKind.SOLO)
        session.add(tenant)
        session.flush()

        account = WhatsAppAccount(
            tenant_id=tenant.id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id=instance_id,
            chat_id=chat_id,
            display_name=name,
        )
        session.add(account)
        session.commit()

        load_cache()

        return {
            "tenant_id": tenant.id,
            "tenant_name": tenant.name,
            "account_id": account.id,
            "provider": account.provider.value,
            "provider_instance_id": account.provider_instance_id,
        }

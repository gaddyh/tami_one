"""Seed the database with demo data."""

from __future__ import annotations

from sqlmodel import Session

from app.db.engine import engine, init_db
from app.db.models import Tenant, TenantKind, WhatsAppAccount, WhatsAppProvider


def run_seed() -> dict[str, str]:
    init_db()

    with Session(engine) as session:
        tenant = Tenant(name="Gaddy Test", kind=TenantKind.SOLO)
        session.add(tenant)
        session.flush()

        account = WhatsAppAccount(
            tenant_id=tenant.id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id="7700673764",
            phone_number="972546610653",
            display_name="Gaddy Test WhatsApp",
        )
        session.add(account)
        session.commit()

        return {
            "tenant_id": tenant.id,
            "tenant_name": tenant.name,
            "account_id": account.id,
            "provider": account.provider.value,
            "provider_instance_id": account.provider_instance_id,
        }

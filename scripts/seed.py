"""CLI entry point for seeding the database.
locally 
DATABASE_URL="postgresql://tami_one_postgre_user:YOUR_PASSWORD@es.render.com/tami_one_postgre" .venv/bin/python scripts/seed.py

"""

from app.db.seed import run_seed
from app.db.engine import init_db


def main() -> None:
    #result = run_seed(name="Gaddy Test", instance_id="7700673764", chat_id="972546610653@c.us", overwrite=False)
    result = run_seed(name="Irit", instance_id="7700678954", chat_id="972522486836@c.us", overwrite=False)
    print(f"Tenant:   {result['tenant_id']}  ({result['tenant_name']})")
    print(f"Account:  {result['account_id']}  ({result['provider']} / {result['provider_instance_id']})")


if __name__ == "__main__":
    init_db()


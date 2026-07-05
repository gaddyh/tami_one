"""CLI entry point for seeding the database.
locally 
DATABASE_URL="postgresql://tami_one_postgre_user:YOUR_PASSWORD@es.render.com/tami_one_postgre" .venv/bin/python scripts/seed.py

"""

from app.db.seed import run_seed


def main() -> None:
    result = run_seed(overwrite=True)
    print(f"Tenant:   {result['tenant_id']}  ({result['tenant_name']})")
    print(f"Account:  {result['account_id']}  ({result['provider']} / {result['provider_instance_id']})")


if __name__ == "__main__":
    main()

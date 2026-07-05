"""CLI entry point for seeding the database."""

from app.db.seed import run_seed


def main() -> None:
    result = run_seed()
    print(f"Tenant:   {result['tenant_id']}  ({result['tenant_name']})")
    print(f"Account:  {result['account_id']}  ({result['provider']} / {result['provider_instance_id']})")


if __name__ == "__main__":
    main()

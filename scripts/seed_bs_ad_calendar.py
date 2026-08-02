from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.authority.bs_ad_calendar import _MONTH_LENGTHS, lookup  # noqa: E402
from app.authority.writer import connect  # noqa: E402


def main() -> None:
    inserted = 0
    with connect() as conn, conn.cursor() as cur:
        for bs_year, months in _MONTH_LENGTHS.items():
            if bs_year % 10 == 0:
                print(f"seeding BS {bs_year}")
            for bs_month, days in enumerate(months, start=1):
                for bs_day in range(1, days + 1):
                    ad_date, _ = lookup(bs_year, bs_month, bs_day)
                    cur.execute(
                        """
                        INSERT INTO bs_ad_calendar (bs_year, bs_month, bs_day, ad_date)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (bs_year, bs_month, bs_day, ad_date),
                    )
                    inserted += cur.rowcount
    print(f"seeded {inserted} calendar rows")


if __name__ == "__main__":
    main()

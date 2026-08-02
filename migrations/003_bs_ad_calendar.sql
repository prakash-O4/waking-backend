CREATE TABLE IF NOT EXISTS bs_ad_calendar (
    bs_year   int NOT NULL,
    bs_month  int NOT NULL CHECK (bs_month BETWEEN 1 AND 12),
    bs_day    int NOT NULL CHECK (bs_day BETWEEN 1 AND 32),
    ad_date   date NOT NULL,
    source_kind text NOT NULL DEFAULT 'official_panchanga',
    version   text NOT NULL DEFAULT '1.0',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (bs_year, bs_month, bs_day, version)
);

CREATE TABLE IF NOT EXISTS bs_ad_boundary_window (
    bs_year  int NOT NULL,
    bs_month int NOT NULL,
    bs_day   int NOT NULL,
    description text,
    PRIMARY KEY (bs_year, bs_month, bs_day)
);

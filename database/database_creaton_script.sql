-- Table: public.dim_customer

-- DROP TABLE IF EXISTS public.dim_customer;

-- Create the sequence for customer_id if it doesn't already exist
CREATE SEQUENCE IF NOT EXISTS dim_customer_customer_id_seq;

-- Create the dim_customer table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_customer
(
    customer_id VARCHAR(30) PRIMARY KEY NOT NULL DEFAULT nextval('dim_customer_customer_id_seq'::regclass),
    gender VARCHAR(20) COLLATE pg_catalog."default",
    age INT,
    married VARCHAR(20),
    dependents VARCHAR(20),
    number_of_dependents INT,
    month_of_joining INT,
    CONSTRAINT dim_customer_pkey PRIMARY KEY (customer_id)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_customer
    OWNER to postgres;

-- Table: public.dim_status

DROP TABLE IF EXISTS public.dim_status;

-- Create the sequence for status_id if it doesn't already exist
CREATE SEQUENCE IF NOT EXISTS dim_status_status_id_seq;

-- Create the dim_status table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_status
(
    status_id VARCHAR(30) NOT NULL DEFAULT nextval('dim_status_status_id_seq'),
    customer_status VARCHAR(25),
    churn_value INT,
    churn_category VARCHAR(50),
    churn_reason VARCHAR(255),
    offer VARCHAR(50) COLLATE pg_catalog."default",
    CONSTRAINT dim_status_pkey PRIMARY KEY (status_id)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_status
    OWNER to postgres;

-- Table: public.dim_location

-- DROP TABLE IF EXISTS public.dim_location;

-- Create the sequence for location_id if it doesn't already exist
CREATE SEQUENCE IF NOT EXISTS dim_location_location_id_seq;

-- Create the dim_location table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_location
(
    location_id VARCHAR(30) PRIMARY KEY NOT NULL DEFAULT nextval('dim_location_location_id_seq'::regclass),
    zip_code VARCHAR(15),
    state VARCHAR(50),
    county VARCHAR(50),
    timezone VARCHAR(50),
    area_codes VARCHAR(20),
    country VARCHAR(50),
    latitude NUMERIC,
    longitude NUMERIC
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_location
    OWNER to postgres;

-- Table: public.dim_payment

-- DROP TABLE IF EXISTS public.dim_payment;

-- Create the sequence for payment_id if it doesn't already exist
CREATE SEQUENCE IF NOT EXISTS payment_id_seq;

-- Create the dim_payment table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_payment
(
    payment_id INTEGER NOT NULL DEFAULT nextval('payment_id_seq'::regclass),
    payment_method VARCHAR(50) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT dim_payment_pkey PRIMARY KEY (payment_id),
    CONSTRAINT dim_payment_unique_method UNIQUE (payment_method)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_payment
    OWNER to postgres;


-- Table: public.dim_service

-- DROP TABLE IF EXISTS public.dim_service;

-- Create the sequence for primary keys if necessary
CREATE SEQUENCE IF NOT EXISTS service_id_seq;

-- Create the dim_service table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_service
(
    service_id VARCHAR(30) NOT NULL,
    month INTEGER NOT NULL,
    phone_service BOOLEAN,
    multiple_lines BOOLEAN,
    internet_service BOOLEAN,
    internet_type VARCHAR(50) COLLATE pg_catalog."default",
    streaming_data_consumption INTEGER,
    online_security BOOLEAN,
    online_backup BOOLEAN,
    device_protection_plan BOOLEAN,
    premium_tech_support BOOLEAN,
    streaming_tv BOOLEAN,
    streaming_movies BOOLEAN,
    streaming_music BOOLEAN,
    unlimited_data BOOLEAN,
    satisfaction_score INTEGER,
    PRIMARY KEY (service_id, month),
    FOREIGN KEY (month) REFERENCES public.dim_time(month)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_service
    OWNER to postgres;

-- Table: public.dim_time

-- DROP TABLE IF EXISTS public.dim_time;

-- Create the sequence for primary keys if necessary
CREATE SEQUENCE IF NOT EXISTS time_id_seq;

-- Create the dim_time table with the proper syntax
CREATE TABLE IF NOT EXISTS public.dim_time
(
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    PRIMARY KEY (month)
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.dim_time
    OWNER to postgres;

-- Insert data into the dim_time table
INSERT INTO public.dim_time (month, month_name, year, quarter) VALUES
(1, 'January', 2023, 1),
(2, 'February', 2023, 1),
(3, 'March', 2023, 1),
(4, 'April', 2023, 2),
(5, 'May', 2023, 2),
(6, 'June', 2023, 2),
(7, 'July', 2023, 3),
(8, 'August', 2023, 3),
(9, 'September', 2023, 3),
(10, 'October', 2023, 4),
(11, 'November', 2023, 4),
(12, 'December', 2023, 4),
(13, 'January', 2024, 1),
(14, 'February', 2024, 1);

-- Table: public.fact_telecom

-- DROP TABLE IF EXISTS public.fact_telecom;

-- Create the sequence for fact_id if it doesn't already exist
CREATE SEQUENCE IF NOT EXISTS fact_telecom_fact_id_seq;

-- Create the fact_telecom table with the proper syntax
CREATE TABLE IF NOT EXISTS public.fact_telecom
(
    fact_id INTEGER NOT NULL DEFAULT nextval('fact_telecom_fact_id_seq'::regclass),
    customer_id VARCHAR(30),
    location_id VARCHAR(30),
    service_id VARCHAR(30),
    status_id VARCHAR(30),
    month INTEGER,
    payment_id INTEGER,
    arpu NUMERIC,
    roam_ic NUMERIC,
    roam_og NUMERIC,
    loc_og_t2t NUMERIC,
    loc_og_t2m NUMERIC,
    loc_og_t2f NUMERIC,
    loc_og_t2c NUMERIC,
    std_og_t2t NUMERIC,
    std_og_t2m NUMERIC,
    std_og_t2f NUMERIC,
    std_og_t2c NUMERIC,
    isd_og NUMERIC,
    spl_og NUMERIC,
    og_others NUMERIC,
    loc_ic_t2t NUMERIC,
    loc_ic_t2m NUMERIC,
    loc_ic_t2f NUMERIC,
    std_ic_t2t NUMERIC,
    std_ic_t2m NUMERIC,
    std_ic_t2f NUMERIC,
    std_ic_t2o NUMERIC,
    spl_ic NUMERIC,
    isd_ic NUMERIC,
    ic_others NUMERIC,
    total_rech_amt NUMERIC,
    total_rech_data NUMERIC,
    vol_4g NUMERIC,
    vol_5g NUMERIC,
    arpu_5g NUMERIC,
    arpu_4g NUMERIC,
    night_pck_user BOOLEAN,
    fb_user BOOLEAN,
    aug_vbc_5g NUMERIC,
    referred_a_friend BOOLEAN,
    number_of_referrals INTEGER,
    satisfaction_score INTEGER,
	offer VARCHAR(5),
    CONSTRAINT fact_telecom_pkey PRIMARY KEY (fact_id),
    CONSTRAINT fact_telecom_customer_id_fkey FOREIGN KEY (customer_id)
        REFERENCES public.dim_customer (customer_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fact_telecom_location_id_fkey FOREIGN KEY (location_id)
        REFERENCES public.dim_location (location_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fact_telecom_month_fkey FOREIGN KEY (month)
        REFERENCES public.dim_time (month) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fact_telecom_service_id_fkey FOREIGN KEY (service_id, month)
        REFERENCES public.dim_service (service_id, month) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fact_telecom_status_id_fkey FOREIGN KEY (status_id)
        REFERENCES public.dim_status (status_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT fact_telecom_payment_id_fkey FOREIGN KEY (payment_id)
        REFERENCES public.dim_payment (payment_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.fact_telecom
    OWNER to postgres;

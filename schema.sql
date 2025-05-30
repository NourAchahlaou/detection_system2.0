--
-- PostgreSQL database dump
--

-- Dumped from database version 15.12 (Debian 15.12-1.pgdg120+1)
-- Dumped by pg_dump version 15.12 (Debian 15.12-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: actiontype; Type: TYPE; Schema: public; Owner: airbususer
--

CREATE TYPE public.actiontype AS ENUM (
    'LOGIN',
    'LOGOUT',
    'VIEW_PIECE',
    'ANNOTATE_PIECE',
    'UPDATE_PROFILE',
    'CREATE_PROFILE'
);


ALTER TYPE public.actiontype OWNER TO airbususer;

--
-- Name: roletype; Type: TYPE; Schema: public; Owner: airbususer
--

CREATE TYPE public.roletype AS ENUM (
    'TECHNICIAN',
    'ADMIN',
    'AUDITOR'
);


ALTER TYPE public.roletype OWNER TO airbususer;

--
-- Name: shiftday; Type: TYPE; Schema: public; Owner: airbususer
--

CREATE TYPE public.shiftday AS ENUM (
    'MONDAY',
    'TUESDAY',
    'WEDNESDAY',
    'THURSDAY',
    'FRIDAY',
    'SATURDAY',
    'SUNDAY'
);


ALTER TYPE public.shiftday OWNER TO airbususer;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: activities; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.activities (
    id integer NOT NULL,
    user_id integer NOT NULL,
    action_type public.actiontype NOT NULL,
    "timestamp" timestamp without time zone,
    details text
);


ALTER TABLE public.activities OWNER TO airbususer;

--
-- Name: activities_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.activities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.activities_id_seq OWNER TO airbususer;

--
-- Name: activities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.activities_id_seq OWNED BY public.activities.id;


--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO airbususer;

--
-- Name: shifts; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.shifts (
    id integer NOT NULL,
    user_id integer NOT NULL,
    start_time time without time zone NOT NULL,
    end_time time without time zone NOT NULL,
    day_of_week public.shiftday NOT NULL,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.shifts OWNER TO airbususer;

--
-- Name: shifts_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.shifts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.shifts_id_seq OWNER TO airbususer;

--
-- Name: shifts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.shifts_id_seq OWNED BY public.shifts.id;


--
-- Name: tasks; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.tasks (
    id integer NOT NULL,
    title character varying(100) NOT NULL,
    description text,
    assigned_user_id integer,
    status character varying(50) NOT NULL,
    due_date timestamp without time zone,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.tasks OWNER TO airbususer;

--
-- Name: tasks_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.tasks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.tasks_id_seq OWNER TO airbususer;

--
-- Name: tasks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.tasks_id_seq OWNED BY public.tasks.id;


--
-- Name: user_tokens; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.user_tokens (
    id integer NOT NULL,
    user_id integer,
    access_key character varying(250),
    refresh_key character varying(250),
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    expires_at timestamp without time zone NOT NULL
);


ALTER TABLE public.user_tokens OWNER TO airbususer;

--
-- Name: user_tokens_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.user_tokens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_tokens_id_seq OWNER TO airbususer;

--
-- Name: user_tokens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.user_tokens_id_seq OWNED BY public.user_tokens.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.users (
    id integer NOT NULL,
    airbus_id integer,
    name character varying(150),
    email character varying(255),
    password character varying(100),
    is_active boolean,
    verified_at timestamp without time zone,
    updated_at timestamp without time zone,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    activation_code character varying(8),
    activation_code_expires_at timestamp without time zone,
    role public.roletype NOT NULL
);


ALTER TABLE public.users OWNER TO airbususer;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.users_id_seq OWNER TO airbususer;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: work_hours; Type: TABLE; Schema: public; Owner: airbususer
--

CREATE TABLE public.work_hours (
    id integer NOT NULL,
    user_id integer NOT NULL,
    login_time timestamp without time zone NOT NULL,
    logout_time timestamp without time zone,
    total_minutes integer
);


ALTER TABLE public.work_hours OWNER TO airbususer;

--
-- Name: work_hours_id_seq; Type: SEQUENCE; Schema: public; Owner: airbususer
--

CREATE SEQUENCE public.work_hours_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.work_hours_id_seq OWNER TO airbususer;

--
-- Name: work_hours_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: airbususer
--

ALTER SEQUENCE public.work_hours_id_seq OWNED BY public.work_hours.id;


--
-- Name: activities id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.activities ALTER COLUMN id SET DEFAULT nextval('public.activities_id_seq'::regclass);


--
-- Name: shifts id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.shifts ALTER COLUMN id SET DEFAULT nextval('public.shifts_id_seq'::regclass);


--
-- Name: tasks id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.tasks ALTER COLUMN id SET DEFAULT nextval('public.tasks_id_seq'::regclass);


--
-- Name: user_tokens id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.user_tokens ALTER COLUMN id SET DEFAULT nextval('public.user_tokens_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: work_hours id; Type: DEFAULT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.work_hours ALTER COLUMN id SET DEFAULT nextval('public.work_hours_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: activities pk_activities; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.activities
    ADD CONSTRAINT pk_activities PRIMARY KEY (id);


--
-- Name: shifts pk_shifts; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.shifts
    ADD CONSTRAINT pk_shifts PRIMARY KEY (id);


--
-- Name: tasks pk_tasks; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT pk_tasks PRIMARY KEY (id);


--
-- Name: user_tokens pk_user_tokens; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.user_tokens
    ADD CONSTRAINT pk_user_tokens PRIMARY KEY (id);


--
-- Name: users pk_users; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT pk_users PRIMARY KEY (id);


--
-- Name: work_hours pk_work_hours; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.work_hours
    ADD CONSTRAINT pk_work_hours PRIMARY KEY (id);


--
-- Name: users uq_users_airbus_id; Type: CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT uq_users_airbus_id UNIQUE (airbus_id);


--
-- Name: ix_user_tokens_access_key; Type: INDEX; Schema: public; Owner: airbususer
--

CREATE INDEX ix_user_tokens_access_key ON public.user_tokens USING btree (access_key);


--
-- Name: ix_user_tokens_refresh_key; Type: INDEX; Schema: public; Owner: airbususer
--

CREATE INDEX ix_user_tokens_refresh_key ON public.user_tokens USING btree (refresh_key);


--
-- Name: ix_users_email; Type: INDEX; Schema: public; Owner: airbususer
--

CREATE UNIQUE INDEX ix_users_email ON public.users USING btree (email);


--
-- Name: activities fk_activities_user_id_users; Type: FK CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.activities
    ADD CONSTRAINT fk_activities_user_id_users FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: shifts fk_shifts_user_id_users; Type: FK CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.shifts
    ADD CONSTRAINT fk_shifts_user_id_users FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: tasks fk_tasks_assigned_user_id_users; Type: FK CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.tasks
    ADD CONSTRAINT fk_tasks_assigned_user_id_users FOREIGN KEY (assigned_user_id) REFERENCES public.users(id);


--
-- Name: user_tokens fk_user_tokens_user_id_users; Type: FK CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.user_tokens
    ADD CONSTRAINT fk_user_tokens_user_id_users FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: work_hours fk_work_hours_user_id_users; Type: FK CONSTRAINT; Schema: public; Owner: airbususer
--

ALTER TABLE ONLY public.work_hours
    ADD CONSTRAINT fk_work_hours_user_id_users FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--


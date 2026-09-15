-- Generic PostgreSQL schema for wikidata_movies.
-- Import the entity tables before the relationship tables.
-- statement_id preserves the originating Wikidata statement node.

CREATE TABLE movies (
    row_id TEXT UNIQUE NOT NULL,
    movie_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    release_date TIMESTAMPTZ,
    duration NUMERIC
);

CREATE TABLE people (
    row_id TEXT UNIQUE NOT NULL,
    person_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    birth_date TIMESTAMPTZ
);

CREATE TABLE genres (
    row_id TEXT UNIQUE NOT NULL,
    genre_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE countries (
    row_id TEXT UNIQUE NOT NULL,
    country_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE directors (
    row_id TEXT PRIMARY KEY,
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    person_id TEXT NOT NULL REFERENCES people(person_id),
    statement_id TEXT NOT NULL
);

CREATE TABLE cast (
    row_id TEXT PRIMARY KEY,
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    person_id TEXT NOT NULL REFERENCES people(person_id),
    statement_id TEXT NOT NULL
);

CREATE TABLE movie_genres (
    row_id TEXT PRIMARY KEY,
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    genre_id TEXT NOT NULL REFERENCES genres(genre_id),
    statement_id TEXT NOT NULL
);

CREATE TABLE movie_countries (
    row_id TEXT PRIMARY KEY,
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    country_id TEXT NOT NULL REFERENCES countries(country_id),
    statement_id TEXT NOT NULL
);

CREATE TABLE person_citizenships (
    row_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    country_id TEXT NOT NULL REFERENCES countries(country_id),
    statement_id TEXT NOT NULL
);

CREATE INDEX idx_directors_movie ON directors(movie_id);
CREATE INDEX idx_directors_person ON directors(person_id);
CREATE INDEX idx_cast_movie ON cast(movie_id);
CREATE INDEX idx_cast_person ON cast(person_id);
CREATE INDEX idx_movie_genres_movie ON movie_genres(movie_id);
CREATE INDEX idx_movie_genres_genre ON movie_genres(genre_id);
CREATE INDEX idx_movie_countries_movie ON movie_countries(movie_id);
CREATE INDEX idx_movie_countries_country ON movie_countries(country_id);
CREATE INDEX idx_person_citizenships_person ON person_citizenships(person_id);
CREATE INDEX idx_person_citizenships_country ON person_citizenships(country_id);

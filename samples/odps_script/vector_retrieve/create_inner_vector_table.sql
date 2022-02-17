drop TABLE IF EXISTS query_vector_{TIME_STAMP};
create table query_vector_{TIME_STAMP}(
    query_id BIGINT
    ,vector string
);

INSERT OVERWRITE TABLE query_vector_{TIME_STAMP}
VALUES
    (1, '0.1,0.2,-0.4,0.5'),
    (2, '-0.1,0.8,0.4,0.5'),
    (3, '0.59,0.2,0.4,0.15'),
    (10, '0.1,-0.2,0.4,-0.5'),
    (20, '-0.1,-0.2,0.4,0.5'),
    (30, '0.5,0.2,0.43,0.15')
;

desc query_vector_{TIME_STAMP};

drop TABLE IF EXISTS doc_vector_{TIME_STAMP};
create table doc_vector_{TIME_STAMP}(
    doc_id BIGINT
    ,vector string
);

INSERT OVERWRITE TABLE doc_vector_{TIME_STAMP}
VALUES
    (1, '0.1,0.2,0.4,0.5'),
    (2, '-0.1,0.2,0.4,0.5'),
    (3, '0.5,0.2,0.4,0.5'),
    (10, '0.1,0.2,0.4,0.5'),
    (20, '-0.1,-0.2,0.4,0.5'),
    (30, '0.5,0.2,0.43,0.15')
;

desc doc_vector_{TIME_STAMP};

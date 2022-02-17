drop TABLE IF EXISTS query_vector_{TIME_STAMP};
create table query_vector_{TIME_STAMP}(
    query_id BIGINT
    ,vector string
);

INSERT OVERWRITE TABLE query_vector_{TIME_STAMP}
select * from external_query_vector_{TIME_STAMP};

desc query_vector_{TIME_STAMP};
desc external_query_vector_{TIME_STAMP};

drop TABLE IF EXISTS doc_vector_{TIME_STAMP};
create table doc_vector_{TIME_STAMP}(
    doc_id BIGINT
    ,vector string
);

INSERT OVERWRITE TABLE doc_vector_{TIME_STAMP}
select * from external_doc_vector_{TIME_STAMP};

desc doc_vector_{TIME_STAMP};
desc external_doc_vector_{TIME_STAMP};

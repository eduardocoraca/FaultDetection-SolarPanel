DROP DATABASE IF EXISTS db;
CREATE DATABASE db;
USE db;

CREATE USER 'usuario'@'%' IDENTIFIED BY 'senha_usuario';
GRANT ALL PRIVILEGES ON db.* to 'usuario'@'%';

CREATE TABLE paineis(
    id_painel INT NOT NULL AUTO_INCREMENT,
    painel VARCHAR(50) NOT NULL,
    status INT,
    img LONGBLOB,
    num_celulas_ng INT,
    data_hora DATETIME,
    comentarios VARCHAR(255),
    crit_sf float,
    crit_tr float,
    PRIMARY KEY (id_painel)
    );

CREATE TABLE celulas (
    id_painel INT NOT NULL,
    local VARCHAR(50) NOT NULL,
    painel VARCHAR(50) NOT NULL,
    trinca INT,
    solda_fria INT,
    outros INT,
    CONSTRAINT FK_id_painel FOREIGN KEY (id_painel) REFERENCES paineis(id_painel)
);

CREATE TABLE celulas_deteccao (
    id_painel INT NOT NULL,
    local VARCHAR(50) NOT NULL,
    painel VARCHAR(50) NOT NULL,
    status VARCHAR(50),
    tamanho float,
    tempo float,
    CONSTRAINT celulas_deteccao_FK FOREIGN KEY (id_painel) REFERENCES paineis(id_painel)
);

CREATE TABLE celulas_segmentacao (
    id_painel INT NOT NULL,
    local VARCHAR(50) NOT NULL,
    painel VARCHAR(50) NOT NULL,
    status VARCHAR(50),
    tamanho float,
    tempo float,
    CONSTRAINT celulas_segmentacao_FK FOREIGN KEY (id_painel) REFERENCES paineis(id_painel)
);

CREATE TABLE celulas_vit (
    id_painel INT NOT NULL,
    local VARCHAR(50) NOT NULL,
    painel VARCHAR(50) NOT NULL,
    status VARCHAR(50),
    tamanho float,
    tempo float,
    CONSTRAINT celulas_vit_FK FOREIGN KEY (id_painel) REFERENCES paineis(id_painel)
);
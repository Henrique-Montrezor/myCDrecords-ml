# Recomendações PyTorch

## Modelo experimental servindo o feed

`mycdrecords_ml.serving` treina BPR/PyTorch em CPU com **24 perfis fictícios e
72 interações simuladas entre 24 artistas**. São afinidades de autoria própria,
não histórico de ouvintes, tags ou estatísticas de fãs. Nenhuma conta/review falsa
é inserida no MySQL. Este fluxo não lê `.env`, usuários ou provedores.

O artefato versionado [artist-bpr-v1.json](models/artist-bpr-v1.json) contém apenas
vetores de artistas (4 dimensões), protocolo, métricas e uma prova de paridade.
Não contém embeddings de usuários. A API inclui uma cópia e executa inferência
Node sem depender de Python no servidor HTTP.

Para retreinar e publicar as duas cópias, com os repositórios como pastas irmãs:

```powershell
.\.venv\Scripts\python.exe -m mycdrecords_ml.serving --output .\models\artist-bpr-v1.json --output ..\myCDrecords-API\src\modules\recommendations\models\artist-bpr-v1.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Resultado verificado: 200 épocas, seed 42, perda **0,69207 → 0,03232**.
Validação: 36 perfis novos com dois artistas observados; teste: 24 perfis novos com
um artista observado, diferentes dos três positivos por perfil de treino.
Recall@3 multi-alvo **1,0**, contra **0,16667** da popularidade empatada por índice.
O corte de afinidade 0,35 também é aplicado na avaliação. Todos compartilham os
mesmos blocos fictícios: **isto testa generalização mecânica do fold-in, não
relevância para pessoas reais**. Não houve ajuste de épocas pelo teste.

Na API, executar `npm run ml:catalog:seed` e `npm run ml:catalog:verify`.
O bootstrap consulta o browse oficial MusicBrainz com MBIDs independentes
referenciados por Wikidata/P434, valida créditos exatos e guarda somente campos
core/CC0. Não depende da busca textual, que retornou 503 nesta homologação;
respeita o cliente existente de fila/backoff. Registros existentes não são
sobrescritos. Foram consultados **244 álbuns** no MySQL; dois perfis fictícios
produziram rankings diferentes de 10 itens cada, inclusive pelo build compilado.

No aplicativo, abrir `http://localhost:5173/recommendations?engine=ml`,
selecionar artistas suportados em **Ajuste seu gosto**, adicionar e salvar.
O seletor não depende de busca externa. O link **Ver minhas sugestões ML no feed**
abre `http://localhost:5173/?engine=ml`.

- Escolha explícita na URL; reload/voltar preservam o modo, navegação sem o parâmetro volta ao motor original.
- Inferência privada somente com notas/favoritos próprios ligados ao catálogo MusicBrainz e artistas escolhidos. Nota explícita prevalece sobre favorito; 3 é neutra, 1/2 afastam.
- Sem tags, gêneros associados, texto, capas, terceiros ou Spotify nas features ML; nomes exatos, sem aliases.
- Vetor do perfil e ranking ML não são persistidos ou cacheados. Não retreina com o perfil real.
- Afinidade mínima 0,35 e até dois álbuns por artista; avaliados/favoritos/dispensados ficam de fora.
- Falta de sinais/candidatos informa fallback ao motor original. Falha de artefato, checksum, paridade ou leitura retorna erro, não fallback disfarçado.
- O piloto de coleta consentida abaixo continua separado; seus pesos reais não foram publicados.

Verificação do incremento: **23 testes Python**, **385 testes API** (8 legados
ignorados), **132 testes client**, tipos/lint/build aprovados. UI conferida em
Edge headless isolado com APIs fictícias, preview de produção em localhost:4173:
375px/1280px sem overflow, alvo 44px, foco/ativação por teclado, feed, loading,
vazio, fallback, erro/retry e anônimo. Nenhuma conta real foi lida pelo navegador.

## Laboratório separado

O laboratório visual continua **somente sintético e de desenvolvimento** do [roadmap ML](../ML-ROADMAP.md). Treina embeddings reais com BPR em CPU, salva pesos e publica um snapshot que pode ser inspecionado na aplicação React. O treino privado real tem entrada separada, descrita abaixo.

O laboratório sintético não consulta MySQL, Redis, Spotify, MusicBrainz ou usuários reais; não lê `.env` nem altera as recomendações atuais. O sucesso deste piloto não mede relevância para pessoas reais e não libera promoção para produção.

## Treino privado real e homologação MySQL

Na pasta da API:

```powershell
npm run ml:mysql:diagnose
npm run ml:mysql:verify
npm run ml:pilot:migrate
npm run ml:train:real
```

- `diagnose` verifica DNS, TCP e a saudação do protocolo, sem autenticar nem imprimir endpoint/credenciais.
- `verify` cria um schema único `mycdrecords_ml_tests_<uuid>` no servidor configurado. Exercita migrações 008/009 repetidas, lock entre conexões, BIGINT, precisão de microssegundos, comprovantes, captura, conflitos, rollback, filtros reais do exportador, retirada e expiração com fixtures descartáveis. Não altera registros da aplicação; mantém o schema e seus avisos públicos, mas reverte as linhas de teste. Exige permissão para criar schema e pool com ao menos duas conexões.
- `migrate` aplica somente as migrações aditivas ML no banco da aplicação. Não convida, adere, faz backfill nem liga coleta.
- `train` consulta somente eventos ainda válidos de contas ativas, com comprovante correspondente à versão **`ml-pilot-v2`** e ledger independente core/CC0. Aceites antigos não são convertidos. Convites, adesão própria e coleta prospectiva continuam necessários; nenhum comando fabrica dados ou autorização.

O Node gera pseudônimos HMAC diferentes por execução para usuários e álbuns e envia somente o contrato fechado `schemaVersion: 2` pelo stdin para `python -m mycdrecords_ml.real`. O filho não herda credenciais de banco/provedores. IDs, notas e horários não são impressos nem gravados em datasets. O contrato sintético anterior continua recusando dados reais.

O PyTorch reaproveita o replay temporal e BPR em CPU. Cortes globais por tempo observado: 60%/20%/20% da extensão temporal, não de linhas aleatórias; ocorrência, observação e sequência decidem o estado conhecido. Seleciona 50/100/200 épocas somente na validação e mede teste contra popularidade. Exige positivos treináveis, contraste e alvos quentes nas duas janelas; isso é viabilidade mecânica, não tamanho amostral suficiente para qualidade. Limites deste ensaio: 100 usuários, 200 álbuns, 5.000 eventos, 2 MiB e 100.000 triplas; entradas maiores falham explicitamente.

**Treino real em memória, sem publicação:** o processo termina sem salvar pesos, mapas, datasets ou rankings individuais. O Node reconsulta autorização, geração e linhas elegíveis antes de aceitar o resultado; retirada, expiração ou exclusão invalidam a execução. Somente resultados agregados privados são salvos em `artifacts/train-real-result.json`; falha substitui o resultado atual por um bloqueio explícito. O laboratório visual e as recomendações da aplicação não consomem esses resultados. Persistência de modelos/serving continuam fora desta etapa.

**Execução em 16/11/2026:** a conexão inicialmente falhou (`ETIMEDOUT`/`MYSQL_GREETING_TIMEOUT`), mas voltou a responder. A homologação MySQL real passou nos **14 checks**; migrações 008/009 foram aplicadas no banco configurado. O treino real foi executado e retornou **`NO_ELIGIBLE_EVENTS`**: zero eventos autorizados elegíveis, portanto nenhum modelo real foi treinado. Não foram fabricados aceites nem importadas notas antigas. TLS não foi relaxado e `.env` não foi alterado. O documento confidencial não é requisito do comando.

**Verificação do código:** 20 testes Python aprovados, incluindo a ponte Node → PyTorch com fixture descartável; 376 testes/38 suítes da API aprovados, tipos de API/testes e build aprovados. Três timeouts de imagens ocorreram na primeira suíte paralela à carga Python; o arquivo passou isoladamente e a suíte completa passou na repetição sem essa carga. A homologação MySQL foi executada pelo comando privado separado, não simulada pelos testes Jest.

## Ambiente validado

- Windows, Python 3.14.7, PyTorch 2.15.0+cpu e NumPy 2.4.4.
- Ambiente virtual próprio em `.venv`; não é necessário ativá-lo.
- Dependências diretas fixadas em `pyproject.toml`; NumPy também atende à interoperabilidade esperada pelo runtime PyTorch.
- Testes pela biblioteca padrão `unittest`, sem instalar um segundo runner.
- Portas locais: laboratório Express em 3005 e Vite em 5173. Não é necessário iniciar a API principal nem conectar uma conta.

Outras versões/plataformas, lock completo de dependências transitivas e CI Python continuam pendentes de validação; não presumir determinismo entre sistemas.

## Preparar e treinar

Em PowerShell, nesta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m mycdrecords_ml train --synthetic
```

O treino sintético usa 200 épocas, seed 42 e CPU, sem GPU. A flag `--synthetic` é obrigatória nesse comando; dados reais só entram pelo fluxo privado separado acima.

Na raiz do workspace, também estão disponíveis as tarefas do VS Code:

1. `ML: install CPU dependencies` (depois de criar `.venv`).
2. `ML: test synthetic PyTorch`.
3. `ML: train synthetic snapshot`.

## Abrir na aplicação

Em um terminal separado, na pasta da API:

```powershell
npm run ml:lab
```

Em outro terminal, na pasta do client:

```powershell
npm run dev
```

Abrir **http://localhost:5173/recommendations/lab**. Se o client já estiver rodando nessa porta, não inicie outra instância.

- Escolher um dos 12 perfis sintéticos.
- Comparar “Ranking PyTorch” com “Referência por popularidade”.
- Conferir positivos de treino, alvo reservado, perda e métricas.
- Recarregar a página: o perfil fica na URL e o resultado continua vindo do snapshot salvo.
- Retreinar manualmente e recarregar para ler a nova publicação.
- A página não faz login, não renova sessão e não envia tokens/cookies ao laboratório.
- O link “Laboratório sintético” também aparece em “Para você” apenas em desenvolvimento.

`npm run ml:lab` é uma entrada Express de desenvolvimento do pacote existente, não um novo serviço de produção. Escuta apenas em `localhost:3005`, aceita o client local em 5173/5174 e recusa inicialização fora de desenvolvimento. O endpoint não é montado na API principal.

A rota React e seu chunk não entram no build de produção; requisições ao laboratório pelo cliente HTTP também são bloqueadas nesse modo. Nenhuma flag no arquivo de ambiente precisa ser criada ou alterada.

## Dataset e avaliação

- 12 usuários e 12 discos artificiais, divididos em três famílias de afinidade.
- Cada perfil tem três positivos de treino e um positivo reservado para teste.
- O modelo recebe apenas os positivos de treino; os candidatos de contraste são todos os itens ainda não avaliados nesse treino, inclusive o alvo reservado. O sampler não consulta labels de teste.
- Os candidatos de avaliação excluem os positivos de treino; ranking completo antes do corte em seis itens para exibição.
- Referência: contagem de positivos no treino, com desempate estável por ID.
- Recall@3 e NDCG@3 com um alvo por usuário, calculados sobre todos os perfis.
- Família sintética, ID e alvo de teste não são features do modelo; os índices servem para acessar os embeddings.

É um teste controlado leave-one-out, **não um split temporal real**. Os scores são fatores de ordenação, não probabilidades, notas previstas ou explicações por gênero.

Na execução homologada, a perda caiu de aproximadamente 0,694 para 0,025; NDCG@3/Recall@3 foram 1,0 para o modelo e 0,333 para popularidade. Isso comprova aprendizado do padrão artificial desta fixture, não qualidade comercial ou generalização para o catálogo real.

## Persistência e falhas

- `artifacts/snapshot.json`: único contrato lido pela API, com versão, métricas e rankings sintéticos.
- `artifacts/<snapshotId>.pt`: `state_dict` dos embeddings; recarga testada com `weights_only=True`.
- Snapshot JSON publicado por substituição atômica no mesmo diretório, depois de salvar os pesos.
- A API valida tamanho, schema, identificadores sintéticos, perfis completos, unicidade, ordem e exclusões. Arquivo inválido não vira sucesso vazio.
- Arquivo ausente: HTTP 503 / `ML_LAB_NOT_TRAINED`; inválido: HTTP 503 / `ML_LAB_INVALID_SNAPSHOT`; outros erros de leitura: erro explícito.
- A API nunca executa Python nem treina por requisição. Pesos só são carregados pelo pacote Python e devem vir de fonte confiável.
- Sem cache compartilhado/TTL; o laboratório lê o snapshot a cada GET e responde `Cache-Control: no-store`.
- Artefatos e ambiente virtual são ignorados pelo controle de versão. Manter privados e não usar esta pasta para dados reais.

O publicador é para um operador local por vez; agendamento, coordenação de jobs, rollback transacional e retenção automatizada pertencem às fases futuras. Não executar dois treinos simultâneos nem tratar este processo como pipeline de produção.

## Pipeline temporal sintético

Próxima etapa técnica, ainda sem dados reais e **separada da página do laboratório**:

```powershell
.\.venv\Scripts\python.exe -m mycdrecords_ml temporal --synthetic
```

Também disponível na tarefa VS Code `ML: evaluate synthetic temporal pipeline`. O comando treina, seleciona e avalia em CPU; publica atomicamente `artifacts/temporal-report.json`. `--output` permite escolher outra pasta. Não sobrescreve `snapshot.json`, não publica pesos para serving e não modifica a API, a página ou as sugestões reais. Falhas de validação/treino encerram com código não zero, sem publicar um relatório novo.

### Protocolo congelado `synthetic-temporal-bpr-v1`

- Catálogo artificial com `available_at`; eventos de nota com `occurred_at` e `observed_at`. Nota `None` representa remoção; 1–3 não são positivos; 4–5 são positivos.
- Cortes globais UTC: treino até 03/01/2026, validação em (03/01, 05/01] e teste em (05/01, 07/01], sempre às 00h. O limite direito é inclusivo.
- Replay em cada corte só considera eventos ocorridos **e observados** até aquele instante, ordenados por ocorrência. A última mutação conhecida do par usuário/álbum define o estado; remoção limpa a nota, sem inventar histórico anterior.
- IDs duplicados, mutações simultâneas ambíguas, notas/timestamps inválidos, interação anterior ao catálogo, referência desconhecida e fontes não sintéticas são rejeitados.
- Mapas ordenados de usuários e itens contêm apenas positivos conhecidos no treino. Um item apenas presente no catálogo não recebe recomendação por embedding aleatório; usuários/itens frios são contados separadamente, sem previsão fabricada.
- Cada janela avalia um alvo novo por usuário: entre os estados positivos finais com ocorrência dentro da janela, escolhe o de ocorrência mais antiga e desempata por ID do evento. Exclui itens já avaliados no início da janela, inclusive notas baixas. Eventos antigos que só chegaram depois não viram escolhas novas.
- Ranking completo dos itens quentes elegíveis antes do corte @3, com os mesmos candidatos/exclusões para BPR e popularidade. A popularidade só usa positivos do treino. A cobertura dessas métricas é da coorte quente, não de todo o catálogo.
- Treina três candidatos independentes de 50/100/200 épocas com seed 42. Seleciona por NDCG@3 na validação; empate prefere menos épocas. Mede o teste uma única vez depois da seleção, sem retreino ou ajuste pelo teste.
- No teste, o modelo continua congelado no corte de treino, mas os itens já avaliados até o fim da validação são removidos dos candidatos. Contrastes BPR não consultam positivos futuros; um alvo futuro pode ser contraste.
- Relatório contém hashes do dataset e do recorte de treino, campos, mapas de IDs, cortes, versões Python/PyTorch, configuração, tentativas de validação e métricas finais. Mesmo conteúdo lógico em outra ordem gera os mesmos hashes/mapas.
- Coorte quente de teste vazia gera métricas `null` e contagem zero, não qualidade perfeita/zero inventada; treino ou validação sem evidência impede executar o experimento.

São 12 perfis quentes com dois positivos de treino, um de validação e um de teste; um 13º perfil/álbum chega durante o teste. `coldUsers`/`coldItems` contam identidades distintas com alvos novos, e `unavailableTargets` conta alvos ainda inexistentes no catálogo no início da janela; essas contagens podem se sobrepor.

Na execução validada em Windows/CPU, a validação selecionou 50 épocas. No teste de 12 perfis quentes: Recall@3 1,0 e NDCG@3 aproximadamente 0,803; popularidade 0,333/0,333. Um usuário frio, um item frio e um alvo indisponível foram separados. Resultados artificiais, sem evidência de qualidade para usuários reais.

**Verificação:** 11 testes `unittest` aprovados, incluindo o CLI real, persistência do relatório, rejeição de fonte proibida, limites dos cortes, edição/remoção, chegada tardia e cold start. Alterar somente os rótulos do teste mantém seleção/treino idênticos; os três testes anteriores do laboratório continuam passando. Pylance sem diagnósticos nos arquivos Python alterados.

O relatório não é um dataset exportado nem checkpoint treinado para produção. Não homologa captura real, privacidade, licenças ou operação: a coleta opcional da API está descrita abaixo; lock transitivo/CI, intervalos de confiança, fallback para cold start e serving temporal real continuam pendentes.

## Coleta opcional na API e limites do piloto

Decisões registradas pelo responsável em 16/11/2026:

| Item | Decisão |
| --- | --- |
| Responsável operacional | Administradores - MyCDRecords |
| Finalidade | Treinar e avaliar recomendações de álbuns no piloto interno com novas notas independentes |
| Participantes | Grupo interno com adesão explícita, sem inclusão automática |
| Sinais | Somente notas 1–5 prospectivas, suas alterações e remoções; sem backfill do estado atual |
| Retenção proposta | Máximo de 30 dias para eventos, datasets, pesos e snapshots do piloto |
| Retirada | Bloquear uso, eliminar dados do piloto e invalidar derivados afetados antes de reutilizá-los |
| Base legal e política | Consentimento específico/revogável declarado pelo responsável; política declarada aprovada e confidencial, sem cópia/referência fornecida às ferramentas |
| Comprovantes | Durante participação e por 30 dias após retirada/encerramento; aviso público atual `ml-pilot-v2` separado do cadastro e da versão anterior |

Notas vinculadas a uma conta podem ser dados pessoais; pseudonimização não é anonimização. A intenção de não coletar dados sensíveis não dispensa definir o tratamento aplicável. O código não oferece parecer jurídico nem presume que adesão na interface equivale a uma base legal validada.

**Texto de reviews permanece excluído de toda a coleta ML.** Deste pacote e do futuro dataset também ficam fora nomes, emails, IDs diretos de contas, capas, tags/gêneros, credenciais e conteúdo Spotify. A API guarda IDs internos privados para controlar adesão/retirada; isso não é anonimização nem um contrato de importação Python. Um MBID, sozinho, não comprova proveniência independente.

### Preflight sem banco

Na pasta da API:

```powershell
npm run ml:preflight
```

O comando não lê `.env`, não inicia a API e não consulta MySQL/Redis/provedores. Emite JSON com os motivos de bloqueio e encerra com código **1** enquanto bloqueado; argumentos desconhecidos retornam **2**. Código 1 aqui é uma negativa deliberada, não falha de instalação.

Os registros de governança são declarações, não comprovação automática. O preflight estático cobre `persisted-real-models-and-serving`: `realDataAllowed` e `promotionAllowed` falsos se referem à persistência/publicação de modelos, não ao experimento separado em memória. A ausência de referência compartilhada da política não é bloqueio. A coleta tem gate de runtime **separado e desligado por padrão**; adesão comprovada, proveniência e ativação local estão no [guia da API](../myCDrecords-API/README.md). Não compartilhar documento confidencial, credenciais ou registros com ferramentas de IA.

### Ensaio API → Python sem dados reais

Na raiz do workspace, executar a tarefa VS Code **`ML: rehearse API to PyTorch export`**. Equivalente, na pasta deste pacote:

```powershell
npm --silent --prefix ..\myCDrecords-API run ml:export:rehearsal -- --synthetic |
    .\.venv\Scripts\python.exe -m mycdrecords_ml temporal --synthetic --stdin
```

A API gera exclusivamente sua fixture interna; não aceita arquivos, filtros de usuários ou `--real`. O Python recebe pelo stdin um contrato estrito `schemaVersion: 1`, `mode: synthetic`, `origin: mycdrecords-internal-fixture-v1`, `cuts`, `albums` e `events`. Os campos de catálogo/evento são os do protocolo temporal acima; não há metadados musicais nem campos livres de review.

O consumidor rejeita campos extras, chaves JSON duplicadas, payloads acima de 128 KiB, fontes/modos não sintéticos e inconsistências temporais/referenciais. Só depois de validar executa o treino/avaliação. Entrada inválida retorna código não zero sem substituir o relatório anterior. `--stdin` não é aceito pelo comando `train`.

O ensaio publica apenas `artifacts/temporal-report.json`; não sobrescreve o snapshot servido, não altera dados da aplicação e não habilita ingestão real. Marcar um payload como sintético não comprova sua origem: **não enviar registros reais por esse canal nem renomeá-los para contornar o gate**.

**Homologação:** 15 testes Python aprovados, incluindo execução real do exportador Node, treino PyTorch e erros sem publicação parcial; 66 testes de recomendações da API aprovados, typecheck/build da API aprovados. Pylance sem diagnósticos nos três arquivos Python desta fatia. A tarefa integrada foi executada e produziu os mesmos hashes/métricas do protocolo temporal anterior. Os testes de intercâmbio exigem Node e dependências de desenvolvimento já instaladas no pacote da API; o runtime de treino isolado continua Python.

## Antes de persistir modelos ou publicar recomendações reais

O experimento privado em memória acima não libera persistência de modelos nem serving:

1. Responsável validar o aviso público/versionado contra sua política aprovada, sem precisar expor o documento confidencial às ferramentas. Consentimento específico foi declarado, não certificado juridicamente por este projeto.
2. Homologar no MySQL isolado os controles já implementados na API: adesão/retirada, captura transacional, moderação e proveniência independente. Não importar o estado atual como passado. A adesão é por API; não há novo formulário frontend nesta fatia.
3. Operar retenção durável independente do processo e tratar backups/réplicas. Limpeza de eventos/comprovantes foi implementada; exclusão/invalidação de futuros datasets/pesos/snapshots/caches e restauração sem reintroduzir exclusões ainda precisam de integração antes de persistir modelos.
4. Autorizar explicitamente o ambiente e a coorte; o exportador atual é minimizado e somente em memória. Persistir datasets ou mapas exige novos controles; não reutilizar este ensaio como publicador.
5. Medir cobertura/volume por agregados autorizados e congelar o protocolo real antes de treinar. Pouca evidência impede avaliar qualidade; não transformar o sucesso sintético em promoção.

Spotify continua fora de treinamento, inferência e avaliação. Tags/associações de gêneros MusicBrainz não entram no piloto e exigem validação de licença compatível com uso comercial.

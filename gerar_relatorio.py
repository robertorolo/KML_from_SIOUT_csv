import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import geopandas
from pykml.factory import KML_ElementMaker as KML
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import re
from pretty_html_table import build_table

import warnings
warnings.filterwarnings("ignore")

today = date.today()
ano = today.year
primeiro_dia = date(ano, 1, 1)

#arquivos para plotar os mapas
#shp_path = "estados/estados_2010.shp"
shp_path = "bacias/Bacia_Hidrografica.shp"
#arquivo de nomes
nomes = 'tabelas/nomes.csv'

#lendo o relatorio
numero = input('Numero do relatorio do SIOUT: ')
file_name = 'relatorios/relatorio_{}.csv'.format(numero)
df = pd.read_csv(filepath_or_buffer=file_name, sep=';', encoding= 'unicode_escape')
#filtrando o df
filtro_tipo1 = df['Tipo de Intervenção'] == 'Canalização do curso d\'água'
filtro_tipo2 = df['Tipo de Intervenção'] == 'Cadastro apenas da barragem'
df = df[filtro_tipo1 | filtro_tipo2]

filtro_status1 = df['Status'] == 'Aguardando análise'
filtro_status2 = df['Status'] == 'Aguardando alterações de dados inconsistentes'
filtro_status3 = df['Status'] == 'Concedida'
filtro_status4 = df['Status'] == 'Indeferida'
filtro_status5 = df['Status'] == 'Em análise'
u_status = ['Concedida', 'Indeferida', 'Em análise', 'Aguardando alterações de dados inconsistentes', 'Aguardando análise']
df_filtrado = df[filtro_status1 |filtro_status2 | filtro_status3 | filtro_status4 | filtro_status5]
n_proc = df_filtrado.shape[0]

#verificando dominiliadde
from shapely.geometry import Point, Polygon
print('Verificando a dominialidade...')
f1 = df['Status'] == 'Aguardando formalização de documentos'
f2 = df['Status'] == 'Concluído'
cadastros = df[f1 | f2]
#Lendo geopandas dominialidade
dom = [geopandas.read_file('Dominialidade/Dominialidade_Federal.shp'), geopandas.read_file('Dominialidade/espelhos_dagua_20ha_uniao_RS.shp'), geopandas.read_file('Dominialidade/rios_dominio_uniao_RS.shp'), geopandas.read_file('Dominialidade/rios_dominio_uniao_terras_publicas_RS.shp'), geopandas.read_file('Dominialidade/unidades_conservação_ANA_RS.shp')]
#Loop nas coordenadas dos cadastros
for index, row in cadastros.iterrows():
    point = Point(float(row['Longitude'].replace(',','.')), float(row['Latitude'].replace(',','.')))
    
    #Loop nos doms
    for d in dom:
        #Loop nos shapes
        for index1, row1 in d.iterrows():
            #checando se o ponto pertence ao shape
            if row1['geometry'].contains(point):
                print(row['Número do cadastro']+' '+row['Nome do usuário de água'])
print('\n')
            
#nomes
print('Sincronizando nomes...\n')
df_nomes = df_filtrado[['Número do cadastro', 'Número da portaria', 'Nome do usuário de água', 'Status', 'Data de saída do processo', 'Município']]
df_nomes['Prioridade'] = 'Não'
df_nomes['Nome'] = 'N/D'
df_nomes['AHE'] = 'N/D'
nomes = pd.read_csv("tabelas/nomes.csv", sep=",", encoding='utf8')

for index, row in nomes.iterrows():
    num = row['Número do cadastro']
    name = row['Nome']
    ahe = row['AHE']
    for index1, row1 in df_nomes.iterrows():
        num1 = row1['Número do cadastro']
        if num == num1:
            df_nomes.loc[index1, 'Nome'] = name
            df_nomes.loc[index1, 'AHE'] = ahe

df_filtrado['Nome'] = df_nomes['Nome']
df_filtrado['AHE'] = df_nomes['AHE']

df_nomes = df_nomes[['Prioridade', 'Número do cadastro', 'AHE', 'Nome', 'Nome do usuário de água', 'Município', 'Status', 'Data de saída do processo', 'Número da portaria']]

#gerando arquivos
print('Gerando tabelas... \n')
df_nomes.to_csv('tabelas/nomes_dumped.csv', index=False)
df_nomes.to_excel('tabelas/processos_siout_{}.xlsx'.format(today), index=False, sheet_name='SIOUT')

#aguardando análise
aguardando = df_filtrado[df_filtrado['Status'] == 'Aguardando análise'][['Número do cadastro', 'Nome do usuário de água', 'Formação do responsável técnico']]
print('PROCESSOS AGUARDANDO ANALISE:')
print(aguardando.to_string(index=False))
print('\n')

#em análise
em = df_filtrado[df_filtrado['Status'] == 'Em análise'][['Número do cadastro', 'Nome do usuário de água']]
print('PROCESSOS EM ANALISE:')
print(em.to_string(index=False))
print('\n')

#plotando o mapa
print('Plotando gráficos... \n')
estados = geopandas.read_file(shp_path)

pie_dict = {}
for s in u_status:
    ns = sum(df_filtrado['Status'] == s)
    if ns > 0:
        pie_dict[s] = ns
        
pie_dict_ahe = {}
for s in np.unique(df_filtrado['AHE']):
    ns = sum(df_filtrado['AHE'] == s)
    if ns > 0:
        pie_dict_ahe[s] = ns

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

plt.rcParams['figure.facecolor'] = 'white'
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(15,10))

ax3.pie(pie_dict.values(), autopct=make_autopct(pie_dict.values()), labels=pie_dict.keys())
estados.plot(color='gainsboro', edgecolor='silver', ax=ax1, alpha=1)

ax4.pie(pie_dict_ahe.values(), autopct=make_autopct(pie_dict_ahe.values()), labels=pie_dict_ahe.keys())

filtro_outorga = []
for idx, row in df[['Número da portaria', 'Data de saída do processo']].iterrows():
    if row['Número da portaria'].split('-')[0] == 'O' and row['Data de saída do processo'].split('/')[-1] == str(ano):
        filtro_outorga.append(True)
    else:
        filtro_outorga.append(False)
        
no = []
od = []

dias_unicos = np.unique(df[filtro_outorga]['Data de saída do processo'])
for d in dias_unicos:
    out_dia = np.sum(df[filtro_outorga]['Data de saída do processo'] == d)
    dia = date(int(d.split('/')[-1]), int(d.split('/')[-2]), int(d.split('/')[-3]))
    dias_corridos = dia - primeiro_dia
    no.append(out_dia)
    od.append(dias_corridos.days)

no = np.array(no)
od = np.array(od)

args = np.argsort(od)
no = no[args]
no = np.cumsum(no)
od = od[args]

filtro_rdh = []
for idx, row in df[['Número da portaria', 'Data de saída do processo']].iterrows():
    if row['Número da portaria'].split('-')[0] == 'R' and row['Data de saída do processo'].split('/')[-1] == str(ano):
        filtro_rdh.append(True)
    else:
        filtro_rdh.append(False)
        
nr = []
rd = []

dias_unicos = np.unique(df[filtro_rdh]['Data de saída do processo'])
for d in dias_unicos:
    rdh_dia = np.sum(df[filtro_rdh]['Data de saída do processo'] == d)
    dia = date(int(d.split('/')[-1]), int(d.split('/')[-2]), int(d.split('/')[-3]))
    dias_corridos = dia - primeiro_dia
    nr.append(rdh_dia)
    rd.append(dias_corridos.days)

nr = np.array(nr)
rd = np.array(rd)

args = np.argsort(rd)
nr = nr[args]
nr = np.cumsum(nr)
rd = rd[args]

if len(no) > 0:
    ax2.plot(od, no, label='Outorgas - {}'.format(no[-1]), marker='s')
if len(nr) > 0:
    ax2.plot(rd, nr, label='RDHs - {}'.format(nr[-1]), marker='H')

for s in u_status:
    f = df_filtrado['Status'] == s
    y, x = df_filtrado[f]['Latitude'].values, df_filtrado[f]['Longitude'].values
    x, y = [float(i.replace(',','.')) for i in x], [float(i.replace(',','.')) for i in y]
    ax1.scatter(x, y, label = s, marker='x')
    
def conversion(coord):
    deg, minutes, seconds, direction =  re.split('[°\'"]', coord)
    return (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)

fisicos = pd.read_excel('tabelas/processos_fisicos.xlsx')
xf = []
yf = []
for index, row in fisicos.iterrows():
    if isinstance(row['Latitude'], str):
        lat = conversion(row['Latitude'])
        long = conversion(row['Longitude'])
        xf.append(lat)
        yf.append(long)
    else:
        xf.append(row['Latitude'])
        yf.append(row['Longitude'])

ax1.scatter(yf, xf, label='Processos físicos', color='grey', s=1)

ax1.axis('scaled')
ax1.set_title('Mapa de distribuição')
ax2.set_title('Portarias emitidas no ano de {}'.format(ano))
#ax3.set_aspect('equal')
ax2.set_ylabel('Número de portarias emitidas')
ax2.set_xlabel('Dias corridos')
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')
ax3.set_title('Distribuição por STATUS - Total no SIOUT {}'.format(n_proc))
ax4.set_title('Distribuição por potência - Total no SIOUT {}'.format(n_proc))
ax1.legend(framealpha=0.0)
ax2.legend(framealpha=0.0)
ax2.grid(alpha=0.5, linestyle='--')
ax1.grid(alpha=0.5, linestyle='--')
fig.tight_layout()
plt.savefig('imagens/Status_{}'.format(today), bbox_inches='tight', transparent=False, dpi=100)

#writing kml
print('Gerando arquivo KML... \n')
doc = KML.Document()

icons = {
    'verde':'http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png',
    'amarelo':'http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png',
    'vermelho':'http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png'
}

for color in icons:
    
    s = KML.Style(
            KML.IconStyle(
                KML.scale(1.2),
                KML.Icon(
                    KML.href(icons[color])
                ),
            ),
            id=color,
        )
    
    doc.append(s)

fld_ag_doc = KML.Folder(KML.name('Aguardando formalização de documentos'))
fld_ag_an = KML.Folder(KML.name('Aguardando análise'))
fld_ag_alt = KML.Folder(KML.name('Aguardando alterações de dados inconsistentes'))
fld_an = KML.Folder(KML.name('Em análise'))
fld_conc = KML.Folder(KML.name('Concedida'))
fld_ind = KML.Folder(KML.name('Indeferida'))

for index, row in df_filtrado.iterrows():
    nome=row['AHE']+' '+row['Nome']
    name = row['Número do cadastro']
    usuario = row['Nome do usuário de água']
    status = row['Status']
    corpo_hidrico = row['Corpo Hídrico']
    municipio = row['Município']
    description = '''
Processo: {}
Usuario: {}
Status: {}
Municipio: {}
Corpo Hidrico: {}
    '''.format(name, usuario, status, municipio, corpo_hidrico)
    long = row['Latitude'].replace(',','.')
    lat = row['Longitude'].replace(',','.')
    coordinates = lat+','+long
    
    if status == 'Concedida':
        style = '#verde'
    elif status == 'Indeferida':
        style = '#vermelho'
    else:
        style = '#amarelo'
    
    p = KML.Placemark(
        KML.name(nome),
        KML.Point(KML.coordinates(coordinates)),
        KML.description(description),
        KML.styleUrl(style))
    
    if status == 'Aguardando formalização de documentos':
        fld_ag_doc.append(p)
    elif status == 'Aguardando análise':
        fld_ag_an.append(p)
    elif status == 'Aguardando alterações de dados inconsistentes':
        fld_ag_alt.append(p)
    elif status == 'Em análise':
        fld_an.append(p)
    elif status == 'Concedida':
        fld_conc.append(p)
    elif status == 'Indeferida':
        fld_ind.append(p)

doc.append(fld_ag_doc)
doc.append(fld_ag_an)
doc.append(fld_ag_alt)
doc.append(fld_an)
doc.append(fld_conc)
doc.append(fld_ind)

kml_file_path = 'kml/hidreletricas_SIOUT_{}.kml'.format(today)

kml_str = etree.tostring(doc, pretty_print=True).decode('utf-8')

f = open(kml_file_path, "w")
f.write(kml_str)
f.close()

#monitoramento
print('Lendo tabela de monitoramento... \n')
monitoramento = pd.read_csv('tabelas/monitoramento_simplificado.csv')

ahe = monitoramento['ahe']
labels, count = np.unique(ahe, return_counts=True)

numero = monitoramento['numero']
ncounts = []
for i in labels:
    f = ahe == i
    numerof = numero[f]
    nlabels, ncount = np.unique(numerof, return_counts=True)
    ncounts.append(ncount[0])

emissao = monitoramento['emissao']
ecounts = []
for i in labels:
    f = ahe == i
    emissaof = emissao[f]
    elabels, ecount = np.unique(emissaof, return_counts=True)
    ecounts.append(ecount[1])

fig = plt.figure(figsize=(10,10))
plt.grid(linestyle='--', axis='y')
plt.bar(x=[i for i in range(len(labels))], height=count, color='red', label='não conforme')
plt.bar(x=[i for i in range(len(labels))], height=ncounts, color='green', label='conforme em relação ao número de estações')

for idx, i in enumerate(labels):
    p = round(ncounts[idx]/count[idx]*100, 1)
    plt.annotate(str(p)+'%', (idx, ncounts[idx]))

    
plt.bar(x=[i for i in range(len(labels))], height=ecounts, color='blue', label='conforme em relação a emissão de dados')

for idx, i in enumerate(labels):
    p = round(ecounts[idx]/ncounts[idx]*100, 1)
    plt.annotate(str(p)+'%', (idx, ecounts[idx]))

plt.xticks([i for i in range(len(labels))], labels)
plt.yticks([i for i in range(0, max(count)+10, 5)])
plt.legend()
plt.ylabel('Número de estações')
plt.title('Conformidade em relação as estações')
plt.savefig('imagens/monitoramento_{}'.format(today), transparent=False, dpi=100, bbox_inches='tight')

#generating html files
print('Gerando arquivos html... \n')
t = build_table(df_nomes, color='blue_dark', font_size = '10px')

h = '''
<!DOCTYPE html>
<html lan="pt-br">
	<head>
		<meta charset="UTF-8">
		<title>Hidrelétricas RS</title>
		<link rel="stylesheet" type="text/css" href="style.css">
	</head>

<body>
	<div><img src="header.png" alt="Header" title="Header"></div>

	<ul>
		<li><a href="index.html">Home</a></li>
		<li><a href="#">SIOUT</a>
			<ul>
				<li><a href="siouttabela.html">Tabela</a></li>
				<li><a href="sioutgraficos.html">Gráficos</a></li>
			</ul>
		</li>
		<li><a href="#">Arquivos KML</a>
			<ul>
				<li><a href="kml/hidreletricas_SIOUT_{date}.kml">Processos SIOUT</a></li>
				<li><a href="kml/hidreletricas_fisicos.kml">Processos físicos</a></li>
			</ul>
		</li>
		<li><a href="manual/Manual_SIOUT_hidreletricas.pdf" download>Manual SIOUT</a></li>
		<li><a href="#">Monitoramento</a>
			<ul>
				<li><a href="monitoramentotabela.html">Tabela</a></li>
				<li><a href="monitoramentografico.html">Gráficos</a></li>
			</ul>
		</li>
	</ul>
'''.format(
date=today
)

numerodanoticia = input('Numero da noticia: ')
text_as_string = open('noticias/n{}.txt'.format(numerodanoticia), 'r').read()

home_str = '''
{header}
	
	<h1>Apresentação</h1>
    
    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec scelerisque tortor ultricies diam maximus fringilla. Phasellus in venenatis ex, at ornare quam. Fusce euismod lectus felis, eu consequat massa dignissim non. In venenatis faucibus ante, id rhoncus metus. Fusce vel ullamcorper odio. Nulla in pellentesque leo, fringilla dictum justo. Nunc dignissim ipsum et sapien luctus mattis. Proin at condimentum justo.</p>
    
    <h1>Notícias</h1>
    
    <table style="width:100%">
      <tr>
        <td><img class="left", src="noticias/{img}.jpg" alt="Imagem notícia" title="Imagem notícia", width="200"></td>
        <td>{text}</td>
    </table>
	
	<div class="footer">
	  <p>Secretaria do Meio Ambiente e Infraestrutura - Atualizado em: {date}</p>
	</div>

</body>
</html>
'''.format(
header=h,
date=today,
text=text_as_string,
img='n{}'.format(numerodanoticia)
)

Html_file= open("index.html","w", encoding='utf-8')
Html_file.write(home_str)
Html_file.close()

siouttabela_str = '''
{header}
	
	<h1>Tabela SIOUT</h1>
    
    {content}
	
	<div class="footer">
	  <p>Secretaria do Meio Ambiente e Infraestrutura - Atualizado em: {date}</p>
	</div>

</body>
</html>
'''.format(
header=h,
date=today,
content=t,
)

Html_file= open("siouttabela.html","w", encoding='utf-8')
Html_file.write(siouttabela_str)
Html_file.close()

sioutgrafico_str = '''
{header}
	
	<h1>Gráficos SIOUT</h1>
    
    <img class="center", src="imagens/Status_{date}.png" alt="Gráficos SIOUT" title="Gráficos SIOUT", width="900">
	
	<div class="footer">
	  <p>Secretaria do Meio Ambiente e Infraestrutura - Atualizado em: {date}</p>
	</div>

</body>
</html>
'''.format(
header=h,
date=today,
)

Html_file= open("sioutgraficos.html","w", encoding='utf-8')
Html_file.write(sioutgrafico_str)
Html_file.close()

new_df = pd.DataFrame()
new_df['AHE'] = monitoramento['ahe']
new_df['Nome'] = monitoramento['nome']
new_df['Usuário de água'] = monitoramento['usuario']
new_df['Conformidade em relação ao número'] = monitoramento['numero']
new_df['Conformidade em relação a emissão'] = monitoramento['emissao']

t = build_table(new_df, color='blue_dark', font_size = '10px')

monitoramentotabela_str = '''
{header}
	
	<h1>Tabela Monitoramento</h1>
    
    {content}
	
	<div class="footer">
	  <p>Secretaria do Meio Ambiente e Infraestrutura - Atualizado em: {date}</p>
	</div>

</body>
</html>
'''.format(
header=h,
date=today,
content=t,
)

Html_file= open("monitoramentotabela.html","w", encoding='utf-8')
Html_file.write(monitoramentotabela_str)
Html_file.close()

monitoramentografico_str = '''
{header}
	
	<h1>Gráfico monitoramento</h1>
    
    <img class="center", src="imagens/monitoramento_{date}.png" alt="Gráfico monitoramento" title="Gráficos monitoramento", width="600">
	
	<div class="footer">
	  <p>Secretaria do Meio Ambiente e Infraestrutura - Atualizado em: {date}</p>
	</div>

</body>
</html>
'''.format(
header=h,
date=today,
)

Html_file= open("monitoramentografico.html","w", encoding='utf-8')
Html_file.write(monitoramentografico_str)
Html_file.close()


print('Feito!')

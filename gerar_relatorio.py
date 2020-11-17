import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import geopandas
from pykml.factory import KML_ElementMaker as KML
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import re

import warnings
warnings.filterwarnings("ignore")

today = date.today()

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
aguardando = df_filtrado[df_filtrado['Status'] == 'Aguardando análise'][['Número do cadastro', 'Nome do usuário de água']]
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

def plot_shape(idt, ax, sf):
    shape_ex = sf.shape(idt)
    x_lon = np.zeros((len(shape_ex.points),1))
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    ax.plot(x_lon,y_lat,c='gray') 
    x0 = np.mean(x_lon)
    y0 = np.mean(y_lat)

pie_dict = {}
for s in u_status:
    ns = sum(df_filtrado['Status'] == s)
    if ns > 0:
        pie_dict[s] = ns

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

plt.rcParams['figure.facecolor'] = 'white'
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,10), gridspec_kw={'width_ratios': [1, 0.84]})
#fig.suptitle('Processos de hidrelétricas do SIOUT', size=16)
ax2.pie(pie_dict.values(), autopct=make_autopct(pie_dict.values()), labels=pie_dict.keys())
#plot_shape(22, ax1, sf)
estados.plot(color='gainsboro', edgecolor='silver', ax=ax1, alpha=1)

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
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')
ax2.set_title('Distrubuição por STATUS - Total {}'.format(n_proc))
ax1.legend()
ax1.grid(alpha=0.5, linestyle='--')
fig.tight_layout()
plt.savefig('imagens/Status_{}'.format(today), bbox_inches='tight', transparent=False)

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

print('Feito!')

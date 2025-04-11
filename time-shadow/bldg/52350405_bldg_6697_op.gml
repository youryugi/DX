<?xml version='1.0' encoding='UTF-8'?>
<core:CityModel xmlns:brid="http://www.opengis.net/citygml/bridge/2.0" xmlns:tran="http://www.opengis.net/citygml/transportation/2.0" xmlns:frn="http://www.opengis.net/citygml/cityfurniture/2.0" xmlns:wtr="http://www.opengis.net/citygml/waterbody/2.0" xmlns:sch="http://www.ascc.net/xml/schematron" xmlns:veg="http://www.opengis.net/citygml/vegetation/2.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:tun="http://www.opengis.net/citygml/tunnel/2.0" xmlns:tex="http://www.opengis.net/citygml/texturedsurface/2.0" xmlns:gml="http://www.opengis.net/gml" xmlns:app="http://www.opengis.net/citygml/appearance/2.0" xmlns:gen="http://www.opengis.net/citygml/generics/2.0" xmlns:dem="http://www.opengis.net/citygml/relief/2.0" xmlns:luse="http://www.opengis.net/citygml/landuse/2.0" xmlns:uro="https://www.geospatial.jp/iur/uro/3.0" xmlns:xAL="urn:oasis:names:tc:ciq:xsdschema:xAL:2.0" xmlns:bldg="http://www.opengis.net/citygml/building/2.0" xmlns:smil20="http://www.w3.org/2001/SMIL20/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:smil20lang="http://www.w3.org/2001/SMIL20/Language" xmlns:pbase="http://www.opengis.net/citygml/profiles/base/2.0" xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:grp="http://www.opengis.net/citygml/cityobjectgroup/2.0" xsi:schemaLocation="https://www.geospatial.jp/iur/uro/3.0 ../../schemas/iur/uro/3.0/urbanObject.xsd  http://www.opengis.net/citygml/2.0 http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd http://www.opengis.net/citygml/landuse/2.0 http://schemas.opengis.net/citygml/landuse/2.0/landUse.xsd  http://www.opengis.net/citygml/building/2.0 http://schemas.opengis.net/citygml/building/2.0/building.xsd  http://www.opengis.net/citygml/transportation/2.0 http://schemas.opengis.net/citygml/transportation/2.0/transportation.xsd  http://www.opengis.net/citygml/generics/2.0 http://schemas.opengis.net/citygml/generics/2.0/generics.xsd  http://www.opengis.net/citygml/relief/2.0 http://schemas.opengis.net/citygml/relief/2.0/relief.xsd  http://www.opengis.net/citygml/cityobjectgroup/2.0 http://schemas.opengis.net/citygml/cityobjectgroup/2.0/cityObjectGroup.xsd  http://www.opengis.net/gml http://schemas.opengis.net/gml/3.1.1/base/gml.xsd http://www.opengis.net/citygml/appearance/2.0 http://schemas.opengis.net/citygml/appearance/2.0/appearance.xsd">
	<gml:boundedBy>
		<gml:Envelope srsName="http://www.opengis.net/def/crs/EPSG/0/6697" srsDimension="3">
			<gml:lowerCorner>34.67434548953123 135.56248237261795 0</gml:lowerCorner>
			<gml:upperCorner>34.67510724482295 135.56305339117543 16.74000000001</gml:upperCorner>
		</gml:Envelope>
	</gml:boundedBy>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_6ff860c3-2e29-446d-84fb-94d76a866379">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3001</bldg:class>
			<bldg:measuredHeight uom="m">7.1</bldg:measuredHeight>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.67510724482295 135.5628268229057 0 34.67501923811699 135.5628215018056 0 34.67501930190499 135.56283928857843 0 34.674813463857234 135.56283294976262 0 34.67480755335681 135.56304511609633 0 34.67510192686758 135.56305339117543 0 34.675104741939975 135.56293334056843 0 34.67510724482295 135.5628268229057 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67510724482295 135.5628268229057 1.01240000001 34.675104741939975 135.56293334056843 1.01240000001 34.67510192686758 135.56305339117543 1.01240000001 34.67480755335681 135.56304511609633 1.01240000001 34.674813463857234 135.56283294976262 1.01240000001 34.67501930190499 135.56283928857843 1.01240000001 34.67501923811699 135.5628215018056 1.01240000001 34.67510724482295 135.5628268229057 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67510724482295 135.5628268229057 1.01240000001 34.67501923811699 135.5628215018056 1.01240000001 34.67501923811699 135.5628215018056 8.12 34.67510724482295 135.5628268229057 8.12 34.67510724482295 135.5628268229057 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67501923811699 135.5628215018056 1.01240000001 34.67501930190499 135.56283928857843 1.01240000001 34.67501930190499 135.56283928857843 8.12 34.67501923811699 135.5628215018056 8.12 34.67501923811699 135.5628215018056 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67501930190499 135.56283928857843 1.01240000001 34.674813463857234 135.56283294976262 1.01240000001 34.674813463857234 135.56283294976262 8.12 34.67501930190499 135.56283928857843 8.12 34.67501930190499 135.56283928857843 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674813463857234 135.56283294976262 1.01240000001 34.67480755335681 135.56304511609633 1.01240000001 34.67480755335681 135.56304511609633 8.12 34.674813463857234 135.56283294976262 8.12 34.674813463857234 135.56283294976262 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67480755335681 135.56304511609633 1.01240000001 34.67510192686758 135.56305339117543 1.01240000001 34.67510192686758 135.56305339117543 8.12 34.67480755335681 135.56304511609633 8.12 34.67480755335681 135.56304511609633 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67510192686758 135.56305339117543 1.01240000001 34.675104741939975 135.56293334056843 1.01240000001 34.675104741939975 135.56293334056843 8.12 34.67510192686758 135.56305339117543 8.12 34.67510192686758 135.56305339117543 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.675104741939975 135.56293334056843 1.01240000001 34.67510724482295 135.5628268229057 1.01240000001 34.67510724482295 135.5628268229057 8.12 34.675104741939975 135.56293334056843 8.12 34.675104741939975 135.56293334056843 1.01240000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67510724482295 135.5628268229057 8.12 34.67501923811699 135.5628215018056 8.12 34.67501930190499 135.56283928857843 8.12 34.674813463857234 135.56283294976262 8.12 34.67480755335681 135.56304511609633 8.12 34.67510192686758 135.56305339117543 8.12 34.675104741939975 135.56293334056843 8.12 34.67510724482295 135.5628268229057 8.12</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">2.16</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">123.1</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324635</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_55391817-33d1-458e-af52-a8c54397d472">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3001</bldg:class>
			<bldg:measuredHeight uom="m">10.0</bldg:measuredHeight>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.674598239145865 135.56277077850382 0 34.674739267498396 135.56277964019782 0 34.67474438525798 135.56269842579073 0 34.674602267668256 135.56268749662567 0 34.674598239145865 135.56277077850382 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674598239145865 135.56277077850382 1.33 34.674602267668256 135.56268749662567 1.33 34.67474438525798 135.56269842579073 1.33 34.674739267498396 135.56277964019782 1.33 34.674598239145865 135.56277077850382 1.33</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674598239145865 135.56277077850382 1.33 34.674739267498396 135.56277964019782 1.33 34.674739267498396 135.56277964019782 11.35 34.674598239145865 135.56277077850382 11.35 34.674598239145865 135.56277077850382 1.33</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674739267498396 135.56277964019782 1.33 34.67474438525798 135.56269842579073 1.33 34.67474438525798 135.56269842579073 11.35 34.674739267498396 135.56277964019782 11.35 34.674739267498396 135.56277964019782 1.33</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67474438525798 135.56269842579073 1.33 34.674602267668256 135.56268749662567 1.33 34.674602267668256 135.56268749662567 11.35 34.67474438525798 135.56269842579073 11.35 34.67474438525798 135.56269842579073 1.33</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674602267668256 135.56268749662567 1.33 34.674598239145865 135.56277077850382 1.33 34.674598239145865 135.56277077850382 11.35 34.674602267668256 135.56268749662567 11.35 34.674602267668256 135.56268749662567 1.33</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674598239145865 135.56277077850382 11.35 34.674739267498396 135.56277964019782 11.35 34.67474438525798 135.56269842579073 11.35 34.674602267668256 135.56268749662567 11.35 34.674598239145865 135.56277077850382 11.35</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:specifiedBuildingCoverageRate>0.6</uro:specifiedBuildingCoverageRate>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">1.86</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">113.6</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324722</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_a339667d-c368-43a0-a25b-7dd03168d697">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3002</bldg:class>
			<bldg:measuredHeight uom="m">15.4</bldg:measuredHeight>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.67445065404498 135.5626179094419 0 34.67447129598539 135.56249154603103 0 34.67443114575061 135.56248237261795 0 34.67441166832435 135.5626066565193 0 34.67445065404498 135.5626179094419 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67445065404498 135.5626179094419 1.39380000001 34.67441166832435 135.5626066565193 1.39380000001 34.67443114575061 135.56248237261795 1.39380000001 34.67447129598539 135.56249154603103 1.39380000001 34.67445065404498 135.5626179094419 1.39380000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67445065404498 135.5626179094419 1.39380000001 34.67447129598539 135.56249154603103 1.39380000001 34.67447129598539 135.56249154603103 16.74000000001 34.67445065404498 135.5626179094419 16.74000000001 34.67445065404498 135.5626179094419 1.39380000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67447129598539 135.56249154603103 1.39380000001 34.67443114575061 135.56248237261795 1.39380000001 34.67443114575061 135.56248237261795 16.74000000001 34.67447129598539 135.56249154603103 16.74000000001 34.67447129598539 135.56249154603103 1.39380000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67443114575061 135.56248237261795 1.39380000001 34.67441166832435 135.5626066565193 1.39380000001 34.67441166832435 135.5626066565193 16.74000000001 34.67443114575061 135.56248237261795 16.74000000001 34.67443114575061 135.56248237261795 1.39380000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67441166832435 135.5626066565193 1.39380000001 34.67445065404498 135.5626179094419 1.39380000001 34.67445065404498 135.5626179094419 16.74000000001 34.67441166832435 135.5626066565193 16.74000000001 34.67441166832435 135.5626066565193 1.39380000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67445065404498 135.5626179094419 16.74000000001 34.67447129598539 135.56249154603103 16.74000000001 34.67443114575061 135.56248237261795 16.74000000001 34.67441166832435 135.5626066565193 16.74000000001 34.67445065404498 135.5626179094419 16.74000000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:specifiedBuildingCoverageRate>0.6</uro:specifiedBuildingCoverageRate>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">1.865</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">108</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324804</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_402ff40a-383f-40b1-a01d-650fdfff4c9b">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3001</bldg:class>
			<bldg:measuredHeight uom="m">10.7</bldg:measuredHeight>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.67447129598539 135.56249154603103 0 34.67445065404498 135.5626179094419 0 34.674532577186056 135.56263624788176 0 34.67453037747403 135.5626513183805 0 34.67461392330144 135.56266964834052 0 34.67461838184394 135.56265598455028 0 34.67472222580455 135.56267824538935 0 34.67474117924005 135.56255865608344 0 34.67447129598539 135.56249154603103 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67447129598539 135.56249154603103 1.2994 34.67474117924005 135.56255865608344 1.2994 34.67472222580455 135.56267824538935 1.2994 34.67461838184394 135.56265598455028 1.2994 34.67461392330144 135.56266964834052 1.2994 34.67453037747403 135.5626513183805 1.2994 34.674532577186056 135.56263624788176 1.2994 34.67445065404498 135.5626179094419 1.2994 34.67447129598539 135.56249154603103 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67447129598539 135.56249154603103 1.2994 34.67445065404498 135.5626179094419 1.2994 34.67445065404498 135.5626179094419 11.950000000000001 34.67447129598539 135.56249154603103 11.950000000000001 34.67447129598539 135.56249154603103 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67445065404498 135.5626179094419 1.2994 34.674532577186056 135.56263624788176 1.2994 34.674532577186056 135.56263624788176 11.950000000000001 34.67445065404498 135.5626179094419 11.950000000000001 34.67445065404498 135.5626179094419 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674532577186056 135.56263624788176 1.2994 34.67453037747403 135.5626513183805 1.2994 34.67453037747403 135.5626513183805 11.950000000000001 34.674532577186056 135.56263624788176 11.950000000000001 34.674532577186056 135.56263624788176 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67453037747403 135.5626513183805 1.2994 34.67461392330144 135.56266964834052 1.2994 34.67461392330144 135.56266964834052 11.950000000000001 34.67453037747403 135.5626513183805 11.950000000000001 34.67453037747403 135.5626513183805 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67461392330144 135.56266964834052 1.2994 34.67461838184394 135.56265598455028 1.2994 34.67461838184394 135.56265598455028 11.950000000000001 34.67461392330144 135.56266964834052 11.950000000000001 34.67461392330144 135.56266964834052 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67461838184394 135.56265598455028 1.2994 34.67472222580455 135.56267824538935 1.2994 34.67472222580455 135.56267824538935 11.950000000000001 34.67461838184394 135.56265598455028 11.950000000000001 34.67461838184394 135.56265598455028 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67472222580455 135.56267824538935 1.2994 34.67474117924005 135.56255865608344 1.2994 34.67474117924005 135.56255865608344 11.950000000000001 34.67472222580455 135.56267824538935 11.950000000000001 34.67472222580455 135.56267824538935 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67474117924005 135.56255865608344 1.2994 34.67447129598539 135.56249154603103 1.2994 34.67447129598539 135.56249154603103 11.950000000000001 34.67474117924005 135.56255865608344 11.950000000000001 34.67474117924005 135.56255865608344 1.2994</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67447129598539 135.56249154603103 11.950000000000001 34.67445065404498 135.5626179094419 11.950000000000001 34.674532577186056 135.56263624788176 11.950000000000001 34.67453037747403 135.5626513183805 11.950000000000001 34.67461392330144 135.56266964834052 11.950000000000001 34.67461838184394 135.56265598455028 11.950000000000001 34.67472222580455 135.56267824538935 11.950000000000001 34.67474117924005 135.56255865608344 11.950000000000001 34.67447129598539 135.56249154603103 11.950000000000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:specifiedBuildingCoverageRate>0.6</uro:specifiedBuildingCoverageRate>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">1.86</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">113.6</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324723</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_ef8deb99-22f5-455a-99c8-43c1ab4d4d70">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3001</bldg:class>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.6748043090165 135.56266833113895 0 34.67499121448946 135.56267444169413 0 34.674967129107365 135.56262000665 0 34.674930536219186 135.56262238148528 0 34.67492532917667 135.56260331229214 0 34.674806232609996 135.56260142857863 0 34.6748043090165 135.56266833113895 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.6748043090165 135.56266833113895 1.31230000001 34.674806232609996 135.56260142857863 1.31230000001 34.67492532917667 135.56260331229214 1.31230000001 34.674930536219186 135.56262238148528 1.31230000001 34.674967129107365 135.56262000665 1.31230000001 34.67499121448946 135.56267444169413 1.31230000001 34.6748043090165 135.56266833113895 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.6748043090165 135.56266833113895 1.31230000001 34.67499121448946 135.56267444169413 1.31230000001 34.67499121448946 135.56267444169413 4.31230000001 34.6748043090165 135.56266833113895 4.31230000001 34.6748043090165 135.56266833113895 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67499121448946 135.56267444169413 1.31230000001 34.674967129107365 135.56262000665 1.31230000001 34.674967129107365 135.56262000665 4.31230000001 34.67499121448946 135.56267444169413 4.31230000001 34.67499121448946 135.56267444169413 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674967129107365 135.56262000665 1.31230000001 34.674930536219186 135.56262238148528 1.31230000001 34.674930536219186 135.56262238148528 4.31230000001 34.674967129107365 135.56262000665 4.31230000001 34.674967129107365 135.56262000665 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674930536219186 135.56262238148528 1.31230000001 34.67492532917667 135.56260331229214 1.31230000001 34.67492532917667 135.56260331229214 4.31230000001 34.674930536219186 135.56262238148528 4.31230000001 34.674930536219186 135.56262238148528 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67492532917667 135.56260331229214 1.31230000001 34.674806232609996 135.56260142857863 1.31230000001 34.674806232609996 135.56260142857863 4.31230000001 34.67492532917667 135.56260331229214 4.31230000001 34.67492532917667 135.56260331229214 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674806232609996 135.56260142857863 1.31230000001 34.6748043090165 135.56266833113895 1.31230000001 34.6748043090165 135.56266833113895 4.31230000001 34.674806232609996 135.56260142857863 4.31230000001 34.674806232609996 135.56260142857863 1.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.6748043090165 135.56266833113895 4.31230000001 34.67499121448946 135.56267444169413 4.31230000001 34.674967129107365 135.56262000665 4.31230000001 34.674930536219186 135.56262238148528 4.31230000001 34.67492532917667 135.56260331229214 4.31230000001 34.674806232609996 135.56260142857863 4.31230000001 34.6748043090165 135.56266833113895 4.31230000001</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:specifiedBuildingCoverageRate>0.6</uro:specifiedBuildingCoverageRate>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">1.959</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">118.833</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324656</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
	<core:cityObjectMember>
		<bldg:Building gml:id="bldg_bdb5b52e-3372-436e-9396-ba7dac3411d4">
			<bldg:class codeSpace="../../codelists/Building_class.xml">3001</bldg:class>
			<bldg:measuredHeight uom="m">7.1</bldg:measuredHeight>
			<bldg:lod0FootPrint>
				<gml:MultiSurface>
					<gml:surfaceMember>
						<gml:Polygon>
							<gml:exterior>
								<gml:LinearRing>
									<gml:posList>34.67434548953123 135.56290609822707 0 34.6747295288951 135.5629298445985 0 34.67473253951433 135.56286435499106 0 34.67468907842651 135.5628620735108 0 34.67469144543892 135.5627930953812 0 34.67459784475675 135.5627864942609 0 34.674595167527734 135.5628192451714 0 34.67457091025162 135.56281740842314 0 34.67457184998179 135.56277779191788 0 34.67458708995409 135.56277902130424 0 34.67458981489372 135.56268417927274 0 34.67448151003991 135.5626749275044 0 34.6744813830916 135.562664670642 0 34.67439959156245 135.5626578875709 0 34.67435217258462 135.56273344225542 0 34.67434548953123 135.56290609822707 0 </gml:posList>
								</gml:LinearRing>
							</gml:exterior>
						</gml:Polygon>
					</gml:surfaceMember>
				</gml:MultiSurface>
			</bldg:lod0FootPrint>
			<bldg:lod1Solid>
				<gml:Solid>
					<gml:exterior>
						<gml:CompositeSurface>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67434548953123 135.56290609822707 1.17049999999 34.67435217258462 135.56273344225542 1.17049999999 34.67439959156245 135.5626578875709 1.17049999999 34.6744813830916 135.562664670642 1.17049999999 34.67448151003991 135.5626749275044 1.17049999999 34.67458981489372 135.56268417927274 1.17049999999 34.67458708995409 135.56277902130424 1.17049999999 34.67457184998179 135.56277779191788 1.17049999999 34.67457091025162 135.56281740842314 1.17049999999 34.674595167527734 135.5628192451714 1.17049999999 34.67459784475675 135.5627864942609 1.17049999999 34.67469144543892 135.5627930953812 1.17049999999 34.67468907842651 135.5628620735108 1.17049999999 34.67473253951433 135.56286435499106 1.17049999999 34.6747295288951 135.5629298445985 1.17049999999 34.67434548953123 135.56290609822707 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67434548953123 135.56290609822707 1.17049999999 34.6747295288951 135.5629298445985 1.17049999999 34.6747295288951 135.5629298445985 8.28 34.67434548953123 135.56290609822707 8.28 34.67434548953123 135.56290609822707 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.6747295288951 135.5629298445985 1.17049999999 34.67473253951433 135.56286435499106 1.17049999999 34.67473253951433 135.56286435499106 8.28 34.6747295288951 135.5629298445985 8.28 34.6747295288951 135.5629298445985 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67473253951433 135.56286435499106 1.17049999999 34.67468907842651 135.5628620735108 1.17049999999 34.67468907842651 135.5628620735108 8.28 34.67473253951433 135.56286435499106 8.28 34.67473253951433 135.56286435499106 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67468907842651 135.5628620735108 1.17049999999 34.67469144543892 135.5627930953812 1.17049999999 34.67469144543892 135.5627930953812 8.28 34.67468907842651 135.5628620735108 8.28 34.67468907842651 135.5628620735108 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67469144543892 135.5627930953812 1.17049999999 34.67459784475675 135.5627864942609 1.17049999999 34.67459784475675 135.5627864942609 8.28 34.67469144543892 135.5627930953812 8.28 34.67469144543892 135.5627930953812 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67459784475675 135.5627864942609 1.17049999999 34.674595167527734 135.5628192451714 1.17049999999 34.674595167527734 135.5628192451714 8.28 34.67459784475675 135.5627864942609 8.28 34.67459784475675 135.5627864942609 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.674595167527734 135.5628192451714 1.17049999999 34.67457091025162 135.56281740842314 1.17049999999 34.67457091025162 135.56281740842314 8.28 34.674595167527734 135.5628192451714 8.28 34.674595167527734 135.5628192451714 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67457091025162 135.56281740842314 1.17049999999 34.67457184998179 135.56277779191788 1.17049999999 34.67457184998179 135.56277779191788 8.28 34.67457091025162 135.56281740842314 8.28 34.67457091025162 135.56281740842314 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67457184998179 135.56277779191788 1.17049999999 34.67458708995409 135.56277902130424 1.17049999999 34.67458708995409 135.56277902130424 8.28 34.67457184998179 135.56277779191788 8.28 34.67457184998179 135.56277779191788 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67458708995409 135.56277902130424 1.17049999999 34.67458981489372 135.56268417927274 1.17049999999 34.67458981489372 135.56268417927274 8.28 34.67458708995409 135.56277902130424 8.28 34.67458708995409 135.56277902130424 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67458981489372 135.56268417927274 1.17049999999 34.67448151003991 135.5626749275044 1.17049999999 34.67448151003991 135.5626749275044 8.28 34.67458981489372 135.56268417927274 8.28 34.67458981489372 135.56268417927274 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67448151003991 135.5626749275044 1.17049999999 34.6744813830916 135.562664670642 1.17049999999 34.6744813830916 135.562664670642 8.28 34.67448151003991 135.5626749275044 8.28 34.67448151003991 135.5626749275044 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.6744813830916 135.562664670642 1.17049999999 34.67439959156245 135.5626578875709 1.17049999999 34.67439959156245 135.5626578875709 8.28 34.6744813830916 135.562664670642 8.28 34.6744813830916 135.562664670642 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67439959156245 135.5626578875709 1.17049999999 34.67435217258462 135.56273344225542 1.17049999999 34.67435217258462 135.56273344225542 8.28 34.67439959156245 135.5626578875709 8.28 34.67439959156245 135.5626578875709 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67435217258462 135.56273344225542 1.17049999999 34.67434548953123 135.56290609822707 1.17049999999 34.67434548953123 135.56290609822707 8.28 34.67435217258462 135.56273344225542 8.28 34.67435217258462 135.56273344225542 1.17049999999</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
							<gml:surfaceMember>
								<gml:Polygon>
									<gml:exterior>
										<gml:LinearRing>
											<gml:posList>34.67434548953123 135.56290609822707 8.28 34.6747295288951 135.5629298445985 8.28 34.67473253951433 135.56286435499106 8.28 34.67468907842651 135.5628620735108 8.28 34.67469144543892 135.5627930953812 8.28 34.67459784475675 135.5627864942609 8.28 34.674595167527734 135.5628192451714 8.28 34.67457091025162 135.56281740842314 8.28 34.67457184998179 135.56277779191788 8.28 34.67458708995409 135.56277902130424 8.28 34.67458981489372 135.56268417927274 8.28 34.67448151003991 135.5626749275044 8.28 34.6744813830916 135.562664670642 8.28 34.67439959156245 135.5626578875709 8.28 34.67435217258462 135.56273344225542 8.28 34.67434548953123 135.56290609822707 8.28</gml:posList>
										</gml:LinearRing>
									</gml:exterior>
								</gml:Polygon>
							</gml:surfaceMember>
						</gml:CompositeSurface>
					</gml:exterior>
				</gml:Solid>
			</bldg:lod1Solid>
			<uro:buildingDataQualityAttribute>
				<uro:BuildingDataQualityAttribute>
					<uro:lod1HeightType codeSpace="../../codelists/BuildingDataQualityAttribute_lod1HeightType.xml">2</uro:lod1HeightType>
				</uro:BuildingDataQualityAttribute>
			</uro:buildingDataQualityAttribute>
			<uro:buildingDetailAttribute>
				<uro:BuildingDetailAttribute>
					<uro:surveyYear>2017</uro:surveyYear>
				</uro:BuildingDetailAttribute>
			</uro:buildingDetailAttribute>
			<uro:buildingDisasterRiskAttribute>
				<uro:BuildingRiverFloodingRiskAttribute>
					<uro:description codeSpace="../../codelists/RiverFloodingRiskAttribute_description.xml">2</uro:description>
					<uro:rank codeSpace="../../codelists/RiverFloodingRiskAttribute_rank.xml">2</uro:rank>
					<uro:depth uom="m">1.865</uro:depth>
					<uro:adminType codeSpace="../../codelists/RiverFloodingRiskAttribute_adminType.xml">1</uro:adminType>
					<uro:scale codeSpace="../../codelists/RiverFloodingRiskAttribute_scale.xml">2</uro:scale>
					<uro:duration uom="hour">108</uro:duration>
				</uro:BuildingRiverFloodingRiskAttribute>
			</uro:buildingDisasterRiskAttribute>
			<uro:buildingIDAttribute>
				<uro:BuildingIDAttribute>
					<uro:buildingID>27100-bldg-324727</uro:buildingID>
					<uro:branchID>1</uro:branchID>
					<uro:prefecture codeSpace="../../codelists/Common_localPublicAuthorities.xml">27</uro:prefecture>
					<uro:city codeSpace="../../codelists/Common_localPublicAuthorities.xml">27115</uro:city>
				</uro:BuildingIDAttribute>
			</uro:buildingIDAttribute>
		</bldg:Building>
	</core:cityObjectMember>
</core:CityModel>

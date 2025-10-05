/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.dv.xs;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.impl.Constants;
import com.sun.org.apache.xerces.internal.impl.dv.DatatypeException;
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeFacetException;
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
import com.sun.org.apache.xerces.internal.impl.dv.ValidationContext;
import com.sun.org.apache.xerces.internal.impl.dv.XSFacets;
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
import com.sun.org.apache.xerces.internal.impl.xpath.regex.ParseException;
import com.sun.org.apache.xerces.internal.impl.xpath.regex.RegularExpression;
import com.sun.org.apache.xerces.internal.impl.xs.SchemaSymbols;
import com.sun.org.apache.xerces.internal.impl.xs.util.ObjectListImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.ShortListImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.StringListImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
import com.sun.org.apache.xerces.internal.util.XMLChar;
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
import com.sun.org.apache.xerces.internal.xs.ShortList;
import com.sun.org.apache.xerces.internal.xs.StringList;
import com.sun.org.apache.xerces.internal.xs.XSAnnotation;
import com.sun.org.apache.xerces.internal.xs.XSConstants;
import com.sun.org.apache.xerces.internal.xs.XSFacet;
import com.sun.org.apache.xerces.internal.xs.XSMultiValueFacet;
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
import com.sun.org.apache.xerces.internal.xs.XSObject;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import com.sun.org.apache.xerces.internal.xs.XSSimpleTypeDefinition;
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
import java.math.BigInteger;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.StringTokenizer;
import org.w3c.dom.TypeInfo;

public class XSSimpleTypeDecl implements XSSimpleType, TypeInfo {

    protected static final short DV_STRING;

    protected static final short DV_BOOLEAN;

    protected static final short DV_DECIMAL;

    protected static final short DV_FLOAT;

    protected static final short DV_DOUBLE;

    protected static final short DV_DURATION;

    protected static final short DV_DATETIME;

    protected static final short DV_TIME;

    protected static final short DV_DATE;

    protected static final short DV_GYEARMONTH;

    protected static final short DV_GYEAR;

    protected static final short DV_GMONTHDAY;

    protected static final short DV_GDAY;

    protected static final short DV_GMONTH;

    protected static final short DV_HEXBINARY;

    protected static final short DV_BASE64BINARY;

    protected static final short DV_ANYURI;

    protected static final short DV_QNAME;

    protected static final short DV_PRECISIONDECIMAL;

    protected static final short DV_NOTATION;

    protected static final short DV_ANYSIMPLETYPE;

    protected static final short DV_ID;

    protected static final short DV_IDREF;

    protected static final short DV_ENTITY;

    protected static final short DV_INTEGER;

    protected static final short DV_LIST;

    protected static final short DV_UNION;

    protected static final short DV_YEARMONTHDURATION;

    protected static final short DV_DAYTIMEDURATION;

    protected static final short DV_ANYATOMICTYPE;

    public static final short YEARMONTHDURATION_DT;

    public static final short DAYTIMEDURATION_DT;

    public static final short PRECISIONDECIMAL_DT;

    public static final short ANYATOMICTYPE_DT;

    protected static TypeValidator[] getGDVs();

    protected void setDVs(TypeValidator[] dvs);

    public XSAnnotation lengthAnnotation;

    public XSAnnotation minLengthAnnotation;

    public XSAnnotation maxLengthAnnotation;

    public XSAnnotation whiteSpaceAnnotation;

    public XSAnnotation totalDigitsAnnotation;

    public XSAnnotation fractionDigitsAnnotation;

    public XSObjectListImpl patternAnnotations;

    public XSObjectList enumerationAnnotations;

    public XSAnnotation maxInclusiveAnnotation;

    public XSAnnotation maxExclusiveAnnotation;

    public XSAnnotation minInclusiveAnnotation;

    public XSAnnotation minExclusiveAnnotation;

    public XSSimpleTypeDecl() {
    }

    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, short validateDV, short ordered, boolean bounded, boolean finite, boolean numeric, boolean isImmutable, short builtInKind) {
    }

    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, String uri, short finalSet, boolean isImmutable, XSObjectList annotations, short builtInKind) {
    }

    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, String uri, short finalSet, boolean isImmutable, XSObjectList annotations) {
    }

    protected XSSimpleTypeDecl(String name, String uri, short finalSet, XSSimpleTypeDecl itemType, boolean isImmutable, XSObjectList annotations) {
    }

    protected XSSimpleTypeDecl(String name, String uri, short finalSet, XSSimpleTypeDecl[] memberTypes, XSObjectList annotations) {
    }

    protected XSSimpleTypeDecl setRestrictionValues(XSSimpleTypeDecl base, String name, String uri, short finalSet, XSObjectList annotations);

    protected XSSimpleTypeDecl setListValues(String name, String uri, short finalSet, XSSimpleTypeDecl itemType, XSObjectList annotations);

    protected XSSimpleTypeDecl setUnionValues(String name, String uri, short finalSet, XSSimpleTypeDecl[] memberTypes, XSObjectList annotations);

    public short getType();

    public short getTypeCategory();

    public String getName();

    public String getTypeName();

    public String getNamespace();

    public short getFinal();

    public boolean isFinal(short derivation);

    public XSTypeDefinition getBaseType();

    public boolean getAnonymous();

    public short getVariety();

    public boolean isIDType();

    public short getWhitespace() throws DatatypeException;

    public short getPrimitiveKind();

    public short getBuiltInKind();

    public XSSimpleTypeDefinition getPrimitiveType();

    public XSSimpleTypeDefinition getItemType();

    public XSObjectList getMemberTypes();

    public void applyFacets(XSFacets facets, short presentFacet, short fixedFacet, ValidationContext context) throws InvalidDatatypeFacetException;

    void applyFacets1(XSFacets facets, short presentFacet, short fixedFacet);

    void applyFacets1(XSFacets facets, short presentFacet, short fixedFacet, short patternType);

    void applyFacets(XSFacets facets, short presentFacet, short fixedFacet, short patternType, ValidationContext context) throws InvalidDatatypeFacetException;

    public Object validate(String content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    protected ValidatedInfo getActualEnumValue(String lexical, ValidationContext ctx, ValidatedInfo info) throws InvalidDatatypeValueException;

    public ValidatedInfo validateWithInfo(String content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    public Object validate(Object content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    public void validate(ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    public boolean isEqual(Object value1, Object value2);

    public boolean isIdentical(Object value1, Object value2);

    public static String normalize(String content, short ws);

    protected String normalize(Object content, short ws);

    void reportError(String key, Object[] args) throws InvalidDatatypeFacetException;

    public short getOrdered();

    public boolean getBounded();

    public boolean getFinite();

    public boolean getNumeric();

    public boolean isDefinedFacet(short facetName);

    public short getDefinedFacets();

    public boolean isFixedFacet(short facetName);

    public short getFixedFacets();

    public String getLexicalFacetValue(short facetName);

    public StringList getLexicalEnumeration();

    public ObjectList getActualEnumeration();

    public ObjectList getEnumerationItemTypeList();

    public ShortList getEnumerationTypeList();

    public StringList getLexicalPattern();

    public XSObjectList getAnnotations();

    public boolean derivedFromType(XSTypeDefinition ancestor, short derivation);

    public boolean derivedFrom(String ancestorNS, String ancestorName, short derivation);

    public boolean isDOMDerivedFrom(String ancestorNS, String ancestorName, int derivationMethod);

    static final class ValidationContextImpl implements ValidationContext {

        void setNSContext(NamespaceContext nsContext);

        public boolean needFacetChecking();

        public boolean needExtraChecking();

        public boolean needToNormalize();

        public boolean useNamespaces();

        public boolean isEntityDeclared(String name);

        public boolean isEntityUnparsed(String name);

        public boolean isIdDeclared(String name);

        public void addId(String name);

        public void addIdRef(String name);

        public String getSymbol(String symbol);

        public String getURI(String prefix);

        public Locale getLocale();
    }

    public void reset();

    public XSNamespaceItem getNamespaceItem();

    public void setNamespaceItem(XSNamespaceItem namespaceItem);

    public String toString();

    public XSObjectList getFacets();

    public XSObject getFacet(int facetType);

    public XSObjectList getMultiValueFacets();

    public Object getMinInclusiveValue();

    public Object getMinExclusiveValue();

    public Object getMaxInclusiveValue();

    public Object getMaxExclusiveValue();

    public void setAnonymous(boolean anon);

    private static final class XSFacetImpl implements XSFacet {

        public XSFacetImpl(short kind, String svalue, int ivalue, Object avalue, boolean fixed, XSAnnotation annotation) {
        }

        public XSAnnotation getAnnotation();

        public XSObjectList getAnnotations();

        public short getFacetKind();

        public String getLexicalFacetValue();

        public Object getActualFacetValue();

        public int getIntFacetValue();

        public boolean getFixed();

        public String getName();

        public String getNamespace();

        public XSNamespaceItem getNamespaceItem();

        public short getType();
    }

    private static final class XSMVFacetImpl implements XSMultiValueFacet {

        public XSMVFacetImpl(short kind, StringList svalues, ObjectList avalues, XSObjectList annotations) {
        }

        public short getFacetKind();

        public XSObjectList getAnnotations();

        public StringList getLexicalFacetValues();

        public ObjectList getEnumerationValues();

        public String getName();

        public String getNamespace();

        public XSNamespaceItem getNamespaceItem();

        public short getType();
    }

    private static abstract class AbstractObjectList extends AbstractList<Object> implements ObjectList {

        public Object get(int index);

        public int size();
    }

    public String getTypeNamespace();

    public boolean isDerivedFrom(String typeNamespaceArg, String typeNameArg, int derivationMethod);
}

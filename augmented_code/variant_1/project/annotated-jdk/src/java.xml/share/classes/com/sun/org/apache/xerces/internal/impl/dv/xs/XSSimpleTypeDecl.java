/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.dv.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.impl.Constants;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.DatatypeException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeFacetException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.ValidationContext;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.XSFacets;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xpath.regex.ParseException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xpath.regex.RegularExpression;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.SchemaSymbols;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.ObjectListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.ShortListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.StringListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLChar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
    @Positive
import com.sun.org.apache.xerces.internal.xs.ShortList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSAnnotation;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSConstants;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSFacet;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSMultiValueFacet;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObject;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSSimpleTypeDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.AbstractList;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.StringTokenizer;
    @Positive
import org.w3c.dom.TypeInfo;

    @Positive
public class XSSimpleTypeDecl implements XSSimpleType, TypeInfo {

    @Positive
    protected static final short DV_STRING;

    @Positive
    protected static final short DV_BOOLEAN;

    @Positive
    protected static final short DV_DECIMAL;

    @Positive
    protected static final short DV_FLOAT;

    @Positive
    protected static final short DV_DOUBLE;

    @Positive
    protected static final short DV_DURATION;

    @Positive
    protected static final short DV_DATETIME;

    @Positive
    protected static final short DV_TIME;

    @Positive
    protected static final short DV_DATE;

    @Positive
    protected static final short DV_GYEARMONTH;

    @Positive
    protected static final short DV_GYEAR;

    @Positive
    protected static final short DV_GMONTHDAY;

    @Positive
    protected static final short DV_GDAY;

    @Positive
    protected static final short DV_GMONTH;

    @Positive
    protected static final short DV_HEXBINARY;

    @Positive
    protected static final short DV_BASE64BINARY;

    @Positive
    protected static final short DV_ANYURI;

    @Positive
    protected static final short DV_QNAME;

    @Positive
    protected static final short DV_PRECISIONDECIMAL;

    @Positive
    protected static final short DV_NOTATION;

    @Positive
    protected static final short DV_ANYSIMPLETYPE;

    @Positive
    protected static final short DV_ID;

    @Positive
    protected static final short DV_IDREF;

    @Positive
    protected static final short DV_ENTITY;

    @Positive
    protected static final short DV_INTEGER;

    @Positive
    protected static final short DV_LIST;

    @Positive
    protected static final short DV_UNION;

    @Positive
    protected static final short DV_YEARMONTHDURATION;

    @Positive
    protected static final short DV_DAYTIMEDURATION;

    @Positive
    protected static final short DV_ANYATOMICTYPE;

    @Positive
    public static final short YEARMONTHDURATION_DT;

    @Positive
    public static final short DAYTIMEDURATION_DT;

    @Positive
    public static final short PRECISIONDECIMAL_DT;

    @Positive
    public static final short ANYATOMICTYPE_DT;

    @Positive
    protected static TypeValidator[] getGDVs();

    @Positive
    protected void setDVs(TypeValidator[] dvs);

    @Positive
    public XSAnnotation lengthAnnotation;

    @Positive
    public XSAnnotation minLengthAnnotation;

    @Positive
    public XSAnnotation maxLengthAnnotation;

    @Positive
    public XSAnnotation whiteSpaceAnnotation;

    @Positive
    public XSAnnotation totalDigitsAnnotation;

    @Positive
    public XSAnnotation fractionDigitsAnnotation;

    @Positive
    public XSObjectListImpl patternAnnotations;

    @Positive
    public XSObjectList enumerationAnnotations;

    @Positive
    public XSAnnotation maxInclusiveAnnotation;

    @Positive
    public XSAnnotation maxExclusiveAnnotation;

    @Positive
    public XSAnnotation minInclusiveAnnotation;

    @Positive
    public XSAnnotation minExclusiveAnnotation;

    @Positive
    public XSSimpleTypeDecl() {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, short validateDV, short ordered, boolean bounded, boolean finite, boolean numeric, boolean isImmutable, short builtInKind) {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, String uri, short finalSet, boolean isImmutable, XSObjectList annotations, short builtInKind) {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl(XSSimpleTypeDecl base, String name, String uri, short finalSet, boolean isImmutable, XSObjectList annotations) {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl(String name, String uri, short finalSet, XSSimpleTypeDecl itemType, boolean isImmutable, XSObjectList annotations) {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl(String name, String uri, short finalSet, XSSimpleTypeDecl[] memberTypes, XSObjectList annotations) {
    @Positive
    }

    @Positive
    protected XSSimpleTypeDecl setRestrictionValues(XSSimpleTypeDecl base, String name, String uri, short finalSet, XSObjectList annotations);

    @Positive
    protected XSSimpleTypeDecl setListValues(String name, String uri, short finalSet, XSSimpleTypeDecl itemType, XSObjectList annotations);

    @Positive
    protected XSSimpleTypeDecl setUnionValues(String name, String uri, short finalSet, XSSimpleTypeDecl[] memberTypes, XSObjectList annotations);

    @Positive
    public short getType();

    @Positive
    public short getTypeCategory();

    @Positive
    public String getName();

    @Positive
    public String getTypeName();

    @Positive
    public String getNamespace();

    @Positive
    public short getFinal();

    @Positive
    public boolean isFinal(short derivation);

    @Positive
    public XSTypeDefinition getBaseType();

    @Positive
    public boolean getAnonymous();

    @Positive
    public short getVariety();

    @Positive
    public boolean isIDType();

    @Positive
    public short getWhitespace() throws DatatypeException;

    @Positive
    public short getPrimitiveKind();

    @Positive
    public short getBuiltInKind();

    @Positive
    public XSSimpleTypeDefinition getPrimitiveType();

    @Positive
    public XSSimpleTypeDefinition getItemType();

    @Positive
    public XSObjectList getMemberTypes();

    @Positive
    public void applyFacets(XSFacets facets, short presentFacet, short fixedFacet, ValidationContext context) throws InvalidDatatypeFacetException;

    @Positive
    void applyFacets1(XSFacets facets, short presentFacet, short fixedFacet);

    @Positive
    void applyFacets1(XSFacets facets, short presentFacet, short fixedFacet, short patternType);

    @Positive
    void applyFacets(XSFacets facets, short presentFacet, short fixedFacet, short patternType, ValidationContext context) throws InvalidDatatypeFacetException;

    @Positive
    public Object validate(String content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    @Positive
    protected ValidatedInfo getActualEnumValue(String lexical, ValidationContext ctx, ValidatedInfo info) throws InvalidDatatypeValueException;

    @Positive
    public ValidatedInfo validateWithInfo(String content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    @Positive
    public Object validate(Object content, ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    @Positive
    public void validate(ValidationContext context, ValidatedInfo validatedInfo) throws InvalidDatatypeValueException;

    @Positive
    public boolean isEqual(Object value1, Object value2);

    @Positive
    public boolean isIdentical(Object value1, Object value2);

    @Positive
    public static String normalize(String content, short ws);

    @Positive
    protected String normalize(Object content, short ws);

    @Positive
    void reportError(String key, Object[] args) throws InvalidDatatypeFacetException;

    @Positive
    public short getOrdered();

    @Positive
    public boolean getBounded();

    @Positive
    public boolean getFinite();

    @Positive
    public boolean getNumeric();

    @Positive
    public boolean isDefinedFacet(short facetName);

    @Positive
    public short getDefinedFacets();

    @Positive
    public boolean isFixedFacet(short facetName);

    @Positive
    public short getFixedFacets();

    @Positive
    public String getLexicalFacetValue(short facetName);

    @Positive
    public StringList getLexicalEnumeration();

    @Positive
    public ObjectList getActualEnumeration();

    @Positive
    public ObjectList getEnumerationItemTypeList();

    @Positive
    public ShortList getEnumerationTypeList();

    @Positive
    public StringList getLexicalPattern();

    @Positive
    public XSObjectList getAnnotations();

    @Positive
    public boolean derivedFromType(XSTypeDefinition ancestor, short derivation);

    @Positive
    public boolean derivedFrom(String ancestorNS, String ancestorName, short derivation);

    @Positive
    public boolean isDOMDerivedFrom(String ancestorNS, String ancestorName, int derivationMethod);

    @Positive
    static final class ValidationContextImpl implements ValidationContext {

    @Positive
        void setNSContext(NamespaceContext nsContext);

    @Positive
        public boolean needFacetChecking();

    @Positive
        public boolean needExtraChecking();

    @Positive
        public boolean needToNormalize();

    @Positive
        public boolean useNamespaces();

    @Positive
        public boolean isEntityDeclared(String name);

    @Positive
        public boolean isEntityUnparsed(String name);

    @Positive
        public boolean isIdDeclared(String name);

    @Positive
        public void addId(String name);

    @Positive
        public void addIdRef(String name);

    @Positive
        public String getSymbol(String symbol);

    @Positive
        public String getURI(String prefix);

    @Positive
        public Locale getLocale();
    @Positive
    }

    @Positive
    public void reset();

    @Positive
    public XSNamespaceItem getNamespaceItem();

    @Positive
    public void setNamespaceItem(XSNamespaceItem namespaceItem);

    @Positive
    public String toString();

    @Positive
    public XSObjectList getFacets();

    @Positive
    public XSObject getFacet(int facetType);

    @Positive
    public XSObjectList getMultiValueFacets();

    @Positive
    public Object getMinInclusiveValue();

    @Positive
    public Object getMinExclusiveValue();

    @Positive
    public Object getMaxInclusiveValue();

    @Positive
    public Object getMaxExclusiveValue();

    @Positive
    public void setAnonymous(boolean anon);

    @Positive
    private static final class XSFacetImpl implements XSFacet {

    @Positive
        public XSFacetImpl(short kind, String svalue, int ivalue, Object avalue, boolean fixed, XSAnnotation annotation) {
    @Positive
        }

    @Positive
        public XSAnnotation getAnnotation();

    @Positive
        public XSObjectList getAnnotations();

    @Positive
        public short getFacetKind();

    @Positive
        public String getLexicalFacetValue();

    @Positive
        public Object getActualFacetValue();

    @Positive
        public int getIntFacetValue();

    @Positive
        public boolean getFixed();

    @Positive
        public String getName();

    @Positive
        public String getNamespace();

    @Positive
        public XSNamespaceItem getNamespaceItem();

    @Positive
        public short getType();
    @Positive
    }

    @Positive
    private static final class XSMVFacetImpl implements XSMultiValueFacet {

    @Positive
        public XSMVFacetImpl(short kind, StringList svalues, ObjectList avalues, XSObjectList annotations) {
    @Positive
        }

    @Positive
        public short getFacetKind();

    @Positive
        public XSObjectList getAnnotations();

    @Positive
        public StringList getLexicalFacetValues();

    @Positive
        public ObjectList getEnumerationValues();

    @Positive
        public String getName();

    @Positive
        public String getNamespace();

    @Positive
        public XSNamespaceItem getNamespaceItem();

    @Positive
        public short getType();
    @Positive
    }

    @Positive
    private static abstract class AbstractObjectList extends AbstractList<Object> implements ObjectList {

    @Positive
        public Object get(int index);

    @Positive
        public int size();
    @Positive
    }

    @Positive
    public String getTypeNamespace();

    @Positive
    public boolean isDerivedFrom(String typeNamespaceArg, String typeNameArg, int derivationMethod);
    @Positive
}

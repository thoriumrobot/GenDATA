/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
    @Positive
import com.sun.org.apache.xerces.internal.xs.*;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.models.XSCMValidator;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.models.CMBuilder;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.xs.XSSimpleTypeDecl;
    @Positive
import org.w3c.dom.TypeInfo;

    @Positive
public class XSComplexTypeDecl implements XSComplexTypeDefinition, TypeInfo {

    @Positive
    public XSComplexTypeDecl() {
    @Positive
    }

    @Positive
    public void setValues(String name, String targetNamespace, XSTypeDefinition baseType, short derivedBy, short schemaFinal, short block, short contentType, boolean isAbstract, XSAttributeGroupDecl attrGrp, XSSimpleType simpleType, XSParticleDecl particle, XSObjectListImpl annotations);

    @Positive
    public void setName(String name);

    @Positive
    public short getTypeCategory();

    @Positive
    public String getTypeName();

    @Positive
    public short getFinalSet();

    @Positive
    public String getTargetNamespace();

    @Positive
    @Pure
    @Positive
    public boolean containsTypeID();

    @Positive
    public void setIsAbstractType();

    @Positive
    public void setContainsTypeID();

    @Positive
    public void setIsAnonymous();

    @Positive
    public XSCMValidator getContentModel(CMBuilder cmBuilder);

    @Positive
    public synchronized XSCMValidator getContentModel(CMBuilder cmBuilder, boolean forUPA);

    @Positive
    public XSAttributeGroupDecl getAttrGrp();

    @Positive
    public String toString();

    @Positive
    void appendTypeInfo(StringBuilder str);

    @Positive
    public boolean derivedFromType(XSTypeDefinition ancestor, short derivationMethod);

    @Positive
    public boolean derivedFrom(String ancestorNS, String ancestorName, short derivationMethod);

    @Positive
    public boolean isDOMDerivedFrom(String ancestorNS, String ancestorName, int derivationMethod);

    @Positive
    public void reset();

    @Positive
    public short getType();

    @Positive
    public String getName();

    @Positive
    public boolean getAnonymous();

    @Positive
    public String getNamespace();

    @Positive
    public XSTypeDefinition getBaseType();

    @Positive
    public short getDerivationMethod();

    @Positive
    public boolean isFinal(short derivation);

    @Positive
    public short getFinal();

    @Positive
    public boolean getAbstract();

    @Positive
    public XSObjectList getAttributeUses();

    @Positive
    public XSWildcard getAttributeWildcard();

    @Positive
    public short getContentType();

    @Positive
    public XSSimpleTypeDefinition getSimpleType();

    @Positive
    public XSParticle getParticle();

    @Positive
    public boolean isProhibitedSubstitution(short prohibited);

    @Positive
    public short getProhibitedSubstitutions();

    @Positive
    public XSObjectList getAnnotations();

    @Positive
    public XSNamespaceItem getNamespaceItem();

    @Positive
    void setNamespaceItem(XSNamespaceItem namespaceItem);

    @Positive
    public XSAttributeUse getAttributeUse(String namespace, String name);

    @Positive
    public String getTypeNamespace();

    @Positive
    public boolean isDerivedFrom(String typeNamespaceArg, String typeNameArg, int derivationMethod);
    @Positive
}

// CFWR semantic augmentation - variant 0

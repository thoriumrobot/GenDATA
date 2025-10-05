/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
import com.sun.org.apache.xerces.internal.xs.*;
import com.sun.org.apache.xerces.internal.impl.xs.models.XSCMValidator;
import com.sun.org.apache.xerces.internal.impl.xs.models.CMBuilder;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
import com.sun.org.apache.xerces.internal.impl.dv.xs.XSSimpleTypeDecl;
import org.w3c.dom.TypeInfo;

public class XSComplexTypeDecl implements XSComplexTypeDefinition, TypeInfo {

    public XSComplexTypeDecl() {
    }

    public void setValues(String name, String targetNamespace, XSTypeDefinition baseType, short derivedBy, short schemaFinal, short block, short contentType, boolean isAbstract, XSAttributeGroupDecl attrGrp, XSSimpleType simpleType, XSParticleDecl particle, XSObjectListImpl annotations);

    public void setName(String name);

    public short getTypeCategory();

    public String getTypeName();

    public short getFinalSet();

    public String getTargetNamespace();

    @Pure
    public boolean containsTypeID();

    public void setIsAbstractType();

    public void setContainsTypeID();

    public void setIsAnonymous();

    public XSCMValidator getContentModel(CMBuilder cmBuilder);

    public synchronized XSCMValidator getContentModel(CMBuilder cmBuilder, boolean forUPA);

    public XSAttributeGroupDecl getAttrGrp();

    public String toString();

    void appendTypeInfo(StringBuilder str);

    public boolean derivedFromType(XSTypeDefinition ancestor, short derivationMethod);

    public boolean derivedFrom(String ancestorNS, String ancestorName, short derivationMethod);

    public boolean isDOMDerivedFrom(String ancestorNS, String ancestorName, int derivationMethod);

    public void reset();

    public short getType();

    public String getName();

    public boolean getAnonymous();

    public String getNamespace();

    public XSTypeDefinition getBaseType();

    public short getDerivationMethod();

    public boolean isFinal(short derivation);

    public short getFinal();

    public boolean getAbstract();

    public XSObjectList getAttributeUses();

    public XSWildcard getAttributeWildcard();

    public short getContentType();

    public XSSimpleTypeDefinition getSimpleType();

    public XSParticle getParticle();

    public boolean isProhibitedSubstitution(short prohibited);

    public short getProhibitedSubstitutions();

    public XSObjectList getAnnotations();

    public XSNamespaceItem getNamespaceItem();

    void setNamespaceItem(XSNamespaceItem namespaceItem);

    public XSAttributeUse getAttributeUse(String namespace, String name);

    public String getTypeNamespace();

    public boolean isDerivedFrom(String typeNamespaceArg, String typeNameArg, int derivationMethod);
}

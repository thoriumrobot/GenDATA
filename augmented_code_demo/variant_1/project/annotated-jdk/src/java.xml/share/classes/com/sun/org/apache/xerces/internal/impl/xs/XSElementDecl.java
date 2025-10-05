/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.IdentityConstraint;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMapImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.xni.QName;
    @Positive
import com.sun.org.apache.xerces.internal.xs.ShortList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSAnnotation;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSComplexTypeDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSConstants;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSElementDeclaration;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSValue;

    @Positive
public class XSElementDecl implements XSElementDeclaration {

    @Positive
    public final static short SCOPE_ABSENT;

    @Positive
    public final static short SCOPE_GLOBAL;

    @Positive
    public final static short SCOPE_LOCAL;

    @Positive
    public String fName;

    @Positive
    public String fTargetNamespace;

    @Positive
    public XSTypeDefinition fType;

    @Positive
    public QName fUnresolvedTypeName;

    @Positive
    public short fScope;

    @Positive
    public short fBlock;

    @Positive
    public short fFinal;

    @Positive
    public XSObjectList fAnnotations;

    @Positive
    public ValidatedInfo fDefault;

    @Positive
    public XSElementDecl fSubGroup;

    @Positive
    public void setConstraintType(short constraintType);

    @Positive
    public void setIsNillable();

    @Positive
    public void setIsAbstract();

    @Positive
    public void setIsGlobal();

    @Positive
    public void setIsLocal(XSComplexTypeDecl enclosingCT);

    @Positive
    public void addIDConstraint(IdentityConstraint idc);

    @Positive
    public IdentityConstraint[] getIDConstraints();

    @Positive
    static final IdentityConstraint[] resize(IdentityConstraint[] oldArray, int newSize);

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public void reset();

    @Positive
    public short getType();

    @Positive
    public String getName();

    @Positive
    public String getNamespace();

    @Positive
    public XSTypeDefinition getTypeDefinition();

    @Positive
    public short getScope();

    @Positive
    public XSComplexTypeDefinition getEnclosingCTDefinition();

    @Positive
    public short getConstraintType();

    @Positive
    @Deprecated
    @Positive
    public String getConstraintValue();

    @Positive
    public boolean getNillable();

    @Positive
    public XSNamedMap getIdentityConstraints();

    @Positive
    public XSElementDeclaration getSubstitutionGroupAffiliation();

    @Positive
    public boolean isSubstitutionGroupExclusion(short exclusion);

    @Positive
    public short getSubstitutionGroupExclusions();

    @Positive
    public boolean isDisallowedSubstitution(short disallowed);

    @Positive
    public short getDisallowedSubstitutions();

    @Positive
    public boolean getAbstract();

    @Positive
    public XSAnnotation getAnnotation();

    @Positive
    public XSObjectList getAnnotations();

    @Positive
    public XSNamespaceItem getNamespaceItem();

    @Positive
    void setNamespaceItem(XSNamespaceItem namespaceItem);

    @Positive
    @Deprecated
    @Positive
    public Object getActualVC();

    @Positive
    @Deprecated
    @Positive
    public short getActualVCType();

    @Positive
    @Deprecated
    @Positive
    public ShortList getItemValueTypes();

    @Positive
    public XSValue getValueConstraintValue();
    @Positive
}

// CFWR semantic augmentation - variant 1

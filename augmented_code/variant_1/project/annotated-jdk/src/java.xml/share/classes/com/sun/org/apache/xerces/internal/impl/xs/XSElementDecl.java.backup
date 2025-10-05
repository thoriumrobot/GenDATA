/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
import com.sun.org.apache.xerces.internal.impl.xs.identity.IdentityConstraint;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMapImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
import com.sun.org.apache.xerces.internal.xni.QName;
import com.sun.org.apache.xerces.internal.xs.ShortList;
import com.sun.org.apache.xerces.internal.xs.XSAnnotation;
import com.sun.org.apache.xerces.internal.xs.XSComplexTypeDefinition;
import com.sun.org.apache.xerces.internal.xs.XSConstants;
import com.sun.org.apache.xerces.internal.xs.XSElementDeclaration;
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
import com.sun.org.apache.xerces.internal.xs.XSValue;

public class XSElementDecl implements XSElementDeclaration {

    public final static short SCOPE_ABSENT;

    public final static short SCOPE_GLOBAL;

    public final static short SCOPE_LOCAL;

    public String fName;

    public String fTargetNamespace;

    public XSTypeDefinition fType;

    public QName fUnresolvedTypeName;

    public short fScope;

    public short fBlock;

    public short fFinal;

    public XSObjectList fAnnotations;

    public ValidatedInfo fDefault;

    public XSElementDecl fSubGroup;

    public void setConstraintType(short constraintType);

    public void setIsNillable();

    public void setIsAbstract();

    public void setIsGlobal();

    public void setIsLocal(XSComplexTypeDecl enclosingCT);

    public void addIDConstraint(IdentityConstraint idc);

    public IdentityConstraint[] getIDConstraints();

    static final IdentityConstraint[] resize(IdentityConstraint[] oldArray, int newSize);

    public String toString();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public void reset();

    public short getType();

    public String getName();

    public String getNamespace();

    public XSTypeDefinition getTypeDefinition();

    public short getScope();

    public XSComplexTypeDefinition getEnclosingCTDefinition();

    public short getConstraintType();

    @Deprecated
    public String getConstraintValue();

    public boolean getNillable();

    public XSNamedMap getIdentityConstraints();

    public XSElementDeclaration getSubstitutionGroupAffiliation();

    public boolean isSubstitutionGroupExclusion(short exclusion);

    public short getSubstitutionGroupExclusions();

    public boolean isDisallowedSubstitution(short disallowed);

    public short getDisallowedSubstitutions();

    public boolean getAbstract();

    public XSAnnotation getAnnotation();

    public XSObjectList getAnnotations();

    public XSNamespaceItem getNamespaceItem();

    void setNamespaceItem(XSNamespaceItem namespaceItem);

    @Deprecated
    public Object getActualVC();

    @Deprecated
    public short getActualVCType();

    @Deprecated
    public ShortList getItemValueTypes();

    public XSValue getValueConstraintValue();
}

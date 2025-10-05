/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.dtd;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.util.XMLResourceIdentifierImpl;
import com.sun.org.apache.xerces.internal.xni.XMLResourceIdentifier;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
import java.util.List;

public class XMLDTDDescription extends XMLResourceIdentifierImpl implements com.sun.org.apache.xerces.internal.xni.grammars.XMLDTDDescription {

    protected String fRootName;

    protected List<String> fPossibleRoots;

    public XMLDTDDescription(XMLResourceIdentifier id, String rootName) {
    }

    public XMLDTDDescription(String publicId, String literalId, String baseId, String expandedId, String rootName) {
    }

    public XMLDTDDescription(XMLInputSource source) {
    }

    public String getGrammarType();

    public String getRootName();

    public void setRootName(String rootName);

    public void setPossibleRoots(List<String> possibleRoots);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object desc);

    public int hashCode();
}

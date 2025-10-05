/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import javax.xml.crypto.*;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.*;
import java.security.Provider;
import java.util.*;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

public final class DOMXMLObject extends DOMStructure implements XMLObject {

    public DOMXMLObject(List<? extends XMLStructure> content, String id, String mimeType, String encoding) {
    }

    public DOMXMLObject(Element objElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public List<XMLStructure> getContent();

    public String getId();

    public String getMimeType();

    public String getEncoding();

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @SuppressWarnings("unchecked")
    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();
}

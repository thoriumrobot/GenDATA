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
import javax.xml.crypto.dsig.*;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dom.DOMURIReference;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.*;
import java.util.*;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import org.jcp.xml.dsig.internal.DigesterOutputStream;
import com.sun.org.apache.xml.internal.security.signature.XMLSignatureInput;
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;

public final class DOMReference extends DOMStructure implements Reference, DOMURIReference {

    public static final int MAXIMUM_TRANSFORM_COUNT;

    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> transforms, String id, Provider provider) {
    }

    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> appliedTransforms, Data result, List<? extends Transform> transforms, String id, Provider provider) {
    }

    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> appliedTransforms, Data result, List<? extends Transform> transforms, String id, byte[] digestValue, Provider provider) {
    }

    public DOMReference(Element refElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public DigestMethod getDigestMethod();

    public String getId();

    public String getURI();

    public String getType();

    public List<Transform> getTransforms();

    public byte[] getDigestValue();

    public byte[] getCalculatedDigestValue();

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    public void digest(XMLSignContext signContext) throws XMLSignatureException;

    public boolean validate(XMLValidateContext validateContext) throws XMLSignatureException;

    public Data getDereferencedData();

    public InputStream getDigestInputStream();

    public Node getHere();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    boolean isDigested();
}

/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package org.jcp.xml.dsig.internal.dom;

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
import javax.xml.crypto.*;
    @Positive
import javax.xml.crypto.dsig.*;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dom.DOMURIReference;
    @Positive
import java.io.*;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.security.*;
    @Positive
import java.util.*;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import org.jcp.xml.dsig.internal.DigesterOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.XMLSignatureInput;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;

    @Positive
public final class DOMReference extends DOMStructure implements Reference, DOMURIReference {

    @Positive
    public static final int MAXIMUM_TRANSFORM_COUNT;

    @Positive
    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> transforms, String id, Provider provider) {
    @Positive
    }

    @Positive
    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> appliedTransforms, Data result, List<? extends Transform> transforms, String id, Provider provider) {
    @Positive
    }

    @Positive
    public DOMReference(String uri, String type, DigestMethod dm, List<? extends Transform> appliedTransforms, Data result, List<? extends Transform> transforms, String id, byte[] digestValue, Provider provider) {
    @Positive
    }

    @Positive
    public DOMReference(Element refElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public DigestMethod getDigestMethod();

    @Positive
    public String getId();

    @Positive
    public String getURI();

    @Positive
    public String getType();

    @Positive
    public List<Transform> getTransforms();

    @Positive
    public byte[] getDigestValue();

    @Positive
    public byte[] getCalculatedDigestValue();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    public void digest(XMLSignContext signContext) throws XMLSignatureException;

    @Positive
    public boolean validate(XMLValidateContext validateContext) throws XMLSignatureException;

    @Positive
    public Data getDereferencedData();

    @Positive
    public InputStream getDigestInputStream();

    @Positive
    public Node getHere();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    boolean isDigested();
    @Positive
}

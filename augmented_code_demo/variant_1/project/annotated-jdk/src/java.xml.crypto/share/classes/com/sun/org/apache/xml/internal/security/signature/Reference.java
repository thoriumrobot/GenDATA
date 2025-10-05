/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.signature;

    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Set;
    @Positive
import com.sun.org.apache.xml.internal.security.algorithms.Algorithm;
    @Positive
import com.sun.org.apache.xml.internal.security.algorithms.MessageDigestAlgorithm;
    @Positive
import com.sun.org.apache.xml.internal.security.c14n.CanonicalizationException;
    @Positive
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceData;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceNodeSetData;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceOctetStreamData;
    @Positive
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceSubTreeData;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.InvalidTransformException;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.Transform;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.TransformationException;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.Transforms;
    @Positive
import com.sun.org.apache.xml.internal.security.transforms.params.InclusiveNamespaces;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.Constants;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.DigesterOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolverContext;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolverException;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import org.w3c.dom.Text;

    @Positive
public class Reference extends SignatureElementProxy {

    @Positive
    public static final String OBJECT_URI;

    @Positive
    public static final String MANIFEST_URI;

    @Positive
    public static final int MAXIMUM_TRANSFORM_COUNT;

    @Positive
    protected Reference(Document doc, String baseURI, String referenceURI, Manifest manifest, Transforms transforms, String messageDigestAlgorithm) throws XMLSignatureException {
    @Positive
    }

    @Positive
    protected Reference(Element element, String baseURI, Manifest manifest) throws XMLSecurityException {
    @Positive
    }

    @Positive
    protected Reference(Element element, String baseURI, Manifest manifest, boolean secureValidation) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public MessageDigestAlgorithm getMessageDigestAlgorithm() throws XMLSignatureException;

    @Positive
    public void setURI(String uri);

    @Positive
    public String getURI();

    @Positive
    public void setId(String id);

    @Positive
    public String getId();

    @Positive
    public void setType(String type);

    @Positive
    public String getType();

    @Positive
    public boolean typeIsReferenceToObject();

    @Positive
    public boolean typeIsReferenceToManifest();

    @Positive
    public void generateDigestValue() throws XMLSignatureException, ReferenceNotInitializedException;

    @Positive
    public XMLSignatureInput getContentsBeforeTransformation() throws ReferenceNotInitializedException;

    @Positive
    public XMLSignatureInput getContentsAfterTransformation() throws XMLSignatureException;

    @Positive
    public XMLSignatureInput getNodesetBeforeFirstCanonicalization() throws XMLSignatureException;

    @Positive
    public String getHTMLRepresentation() throws XMLSignatureException;

    @Positive
    public XMLSignatureInput getTransformsOutput();

    @Positive
    public ReferenceData getReferenceData();

    @Positive
    protected XMLSignatureInput dereferenceURIandPerformTransforms(OutputStream os) throws XMLSignatureException;

    @Positive
    public Transforms getTransforms() throws XMLSignatureException, InvalidTransformException, TransformationException, XMLSecurityException;

    @Positive
    public byte[] getReferencedBytes() throws ReferenceNotInitializedException, XMLSignatureException;

    @Positive
    public byte[] getDigestValue() throws XMLSecurityException;

    @Positive
    public boolean verify() throws ReferenceNotInitializedException, XMLSecurityException;

    @Positive
    public String getBaseLocalName();
    @Positive
}

// CFWR semantic augmentation - variant 1

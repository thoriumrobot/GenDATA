/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.signature;

import java.io.IOException;
import java.io.OutputStream;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import com.sun.org.apache.xml.internal.security.algorithms.Algorithm;
import com.sun.org.apache.xml.internal.security.algorithms.MessageDigestAlgorithm;
import com.sun.org.apache.xml.internal.security.c14n.CanonicalizationException;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceData;
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceNodeSetData;
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceOctetStreamData;
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceSubTreeData;
import com.sun.org.apache.xml.internal.security.transforms.InvalidTransformException;
import com.sun.org.apache.xml.internal.security.transforms.Transform;
import com.sun.org.apache.xml.internal.security.transforms.TransformationException;
import com.sun.org.apache.xml.internal.security.transforms.Transforms;
import com.sun.org.apache.xml.internal.security.transforms.params.InclusiveNamespaces;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.DigesterOutputStream;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolver;
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolverContext;
import com.sun.org.apache.xml.internal.security.utils.resolver.ResourceResolverException;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.Text;

public class Reference extends SignatureElementProxy {

    public static final String OBJECT_URI;

    public static final String MANIFEST_URI;

    public static final int MAXIMUM_TRANSFORM_COUNT;

    protected Reference(Document doc, String baseURI, String referenceURI, Manifest manifest, Transforms transforms, String messageDigestAlgorithm) throws XMLSignatureException {
    }

    protected Reference(Element element, String baseURI, Manifest manifest) throws XMLSecurityException {
    }

    protected Reference(Element element, String baseURI, Manifest manifest, boolean secureValidation) throws XMLSecurityException {
    }

    public MessageDigestAlgorithm getMessageDigestAlgorithm() throws XMLSignatureException;

    public void setURI(String uri);

    public String getURI();

    public void setId(String id);

    public String getId();

    public void setType(String type);

    public String getType();

    public boolean typeIsReferenceToObject();

    public boolean typeIsReferenceToManifest();

    public void generateDigestValue() throws XMLSignatureException, ReferenceNotInitializedException;

    public XMLSignatureInput getContentsBeforeTransformation() throws ReferenceNotInitializedException;

    public XMLSignatureInput getContentsAfterTransformation() throws XMLSignatureException;

    public XMLSignatureInput getNodesetBeforeFirstCanonicalization() throws XMLSignatureException;

    public String getHTMLRepresentation() throws XMLSignatureException;

    public XMLSignatureInput getTransformsOutput();

    public ReferenceData getReferenceData();

    protected XMLSignatureInput dereferenceURIandPerformTransforms(OutputStream os) throws XMLSignatureException;

    public Transforms getTransforms() throws XMLSignatureException, InvalidTransformException, TransformationException, XMLSecurityException;

    public byte[] getReferencedBytes() throws ReferenceNotInitializedException, XMLSignatureException;

    public byte[] getDigestValue() throws XMLSecurityException;

    public boolean verify() throws ReferenceNotInitializedException, XMLSecurityException;

    public String getBaseLocalName();
}

/*
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package org.w3c.dom;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public interface Node {

    @Positive
    public static final short ELEMENT_NODE;

    @Positive
    public static final short ATTRIBUTE_NODE;

    @Positive
    public static final short TEXT_NODE;

    @Positive
    public static final short CDATA_SECTION_NODE;

    @Positive
    public static final short ENTITY_REFERENCE_NODE;

    @Positive
    public static final short ENTITY_NODE;

    @Positive
    public static final short PROCESSING_INSTRUCTION_NODE;

    @Positive
    public static final short COMMENT_NODE;

    @Positive
    public static final short DOCUMENT_NODE;

    @Positive
    public static final short DOCUMENT_TYPE_NODE;

    @Positive
    public static final short DOCUMENT_FRAGMENT_NODE;

    @Positive
    public static final short NOTATION_NODE;

    @Positive
    @Pure
    @Positive
    public String getNodeName();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getNodeValue() throws DOMException;

    @Positive
    public void setNodeValue(String nodeValue) throws DOMException;

    @Positive
    @Pure
    @Positive
    public short getNodeType();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Node getParentNode();

    @Positive
    @Pure
    @Positive
    public NodeList getChildNodes();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Node getFirstChild();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Node getLastChild();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Node getPreviousSibling();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Node getNextSibling();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public NamedNodeMap getAttributes();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Document getOwnerDocument();

    @Positive
    public Node insertBefore(Node newChild, @Nullable Node refChild) throws DOMException;

    @Positive
    public Node replaceChild(Node newChild, Node oldChild) throws DOMException;

    @Positive
    public Node removeChild(Node oldChild) throws DOMException;

    @Positive
    public Node appendChild(Node newChild) throws DOMException;

    @Positive
    @Pure
    @Positive
    public boolean hasChildNodes();

    @Positive
    public Node cloneNode(boolean deep);

    @Positive
    public void normalize();

    @Positive
    @Pure
    @Positive
    public boolean isSupported(String feature, @Nullable String version);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getNamespaceURI();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getPrefix();

    @Positive
    public void setPrefix(@Nullable String prefix) throws DOMException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getLocalName();

    @Positive
    @Pure
    @Positive
    public boolean hasAttributes();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getBaseURI();

    @Positive
    public static final short DOCUMENT_POSITION_DISCONNECTED;

    @Positive
    public static final short DOCUMENT_POSITION_PRECEDING;

    @Positive
    public static final short DOCUMENT_POSITION_FOLLOWING;

    @Positive
    public static final short DOCUMENT_POSITION_CONTAINS;

    @Positive
    public static final short DOCUMENT_POSITION_CONTAINED_BY;

    @Positive
    public static final short DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC;

    @Positive
    @Pure
    @Positive
    public short compareDocumentPosition(Node other) throws DOMException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getTextContent() throws DOMException;

    @Positive
    public void setTextContent(String textContent) throws DOMException;

    @Positive
    @Pure
    @Positive
    public boolean isSameNode(Node other);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String lookupPrefix(@Nullable String namespaceURI);

    @Positive
    @Pure
    @Positive
    public boolean isDefaultNamespace(@Nullable String namespaceURI);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String lookupNamespaceURI(@Nullable String prefix);

    @Positive
    @Pure
    @Positive
    public boolean isEqualNode(Node arg);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Object getFeature(String feature, @Nullable String version);

    @Positive
    @Nullable
    @Positive
    public Object setUserData(String key, @Nullable Object data, @Nullable UserDataHandler handler);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Object getUserData(String key);
    @Positive
}

// CFWR semantic augmentation - variant 1

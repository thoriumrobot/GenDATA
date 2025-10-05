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
public interface Document extends Node {

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public DocumentType getDoctype();

    @Positive
    @Pure
    @Positive
    public DOMImplementation getImplementation();

    @Positive
    @Pure
    @Positive
    public Element getDocumentElement();

    @Positive
    public Element createElement(String tagName) throws DOMException;

    @Positive
    public DocumentFragment createDocumentFragment();

    @Positive
    public Text createTextNode(String data);

    @Positive
    public Comment createComment(String data);

    @Positive
    public CDATASection createCDATASection(String data) throws DOMException;

    @Positive
    public ProcessingInstruction createProcessingInstruction(String target, String data) throws DOMException;

    @Positive
    public Attr createAttribute(String name) throws DOMException;

    @Positive
    public EntityReference createEntityReference(String name) throws DOMException;

    @Positive
    @Pure
    @Positive
    public NodeList getElementsByTagName(String tagname);

    @Positive
    public Node importNode(Node importedNode, boolean deep) throws DOMException;

    @Positive
    public Element createElementNS(@Nullable String namespaceURI, String qualifiedName) throws DOMException;

    @Positive
    public Attr createAttributeNS(@Nullable String namespaceURI, String qualifiedName) throws DOMException;

    @Positive
    @Pure
    @Positive
    public NodeList getElementsByTagNameNS(@Nullable String namespaceURI, String localName);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Element getElementById(String elementId);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getInputEncoding();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getXmlEncoding();

    @Positive
    @Pure
    @Positive
    public boolean getXmlStandalone();

    @Positive
    public void setXmlStandalone(boolean xmlStandalone) throws DOMException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getXmlVersion();

    @Positive
    public void setXmlVersion(String xmlVersion) throws DOMException;

    @Positive
    @Pure
    @Positive
    public boolean getStrictErrorChecking();

    @Positive
    public void setStrictErrorChecking(boolean strictErrorChecking);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getDocumentURI();

    @Positive
    public void setDocumentURI(@Nullable String documentURI);

    @Positive
    @Nullable
    @Positive
    public Node adoptNode(@Nullable Node source) throws DOMException;

    @Positive
    @Pure
    @Positive
    public DOMConfiguration getDomConfig();

    @Positive
    public void normalizeDocument();

    @Positive
    public Node renameNode(Node n, @Nullable String namespaceURI, String qualifiedName) throws DOMException;
    @Positive
}

// CFWR semantic augmentation - variant 1

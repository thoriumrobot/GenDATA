/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package org.w3c.dom;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor("nullness")
public interface Document extends Node {

    @Pure
    @Nullable
    public DocumentType getDoctype();

    @Pure
    public DOMImplementation getImplementation();

    @Pure
    public Element getDocumentElement();

    public Element createElement(String tagName) throws DOMException;

    public DocumentFragment createDocumentFragment();

    public Text createTextNode(String data);

    public Comment createComment(String data);

    public CDATASection createCDATASection(String data) throws DOMException;

    public ProcessingInstruction createProcessingInstruction(String target, String data) throws DOMException;

    public Attr createAttribute(String name) throws DOMException;

    public EntityReference createEntityReference(String name) throws DOMException;

    @Pure
    public NodeList getElementsByTagName(String tagname);

    public Node importNode(Node importedNode, boolean deep) throws DOMException;

    public Element createElementNS(@Nullable String namespaceURI, String qualifiedName) throws DOMException;

    public Attr createAttributeNS(@Nullable String namespaceURI, String qualifiedName) throws DOMException;

    @Pure
    public NodeList getElementsByTagNameNS(@Nullable String namespaceURI, String localName);

    @Pure
    @Nullable
    public Element getElementById(String elementId);

    @Pure
    @Nullable
    public String getInputEncoding();

    @Pure
    @Nullable
    public String getXmlEncoding();

    @Pure
    public boolean getXmlStandalone();

    public void setXmlStandalone(boolean xmlStandalone) throws DOMException;

    @Pure
    @Nullable
    public String getXmlVersion();

    public void setXmlVersion(String xmlVersion) throws DOMException;

    @Pure
    public boolean getStrictErrorChecking();

    public void setStrictErrorChecking(boolean strictErrorChecking);

    @Pure
    @Nullable
    public String getDocumentURI();

    public void setDocumentURI(@Nullable String documentURI);

    @Nullable
    public Node adoptNode(@Nullable Node source) throws DOMException;

    @Pure
    public DOMConfiguration getDomConfig();

    public void normalizeDocument();

    public Node renameNode(Node n, @Nullable String namespaceURI, String qualifiedName) throws DOMException;
}

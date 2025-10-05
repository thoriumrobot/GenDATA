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

@AnnotatedFor({ "nullness" })
public interface Element extends Node {

    @Pure
    public String getTagName();

    @Pure
    public String getAttribute(String name);

    public void setAttribute(String name, String value) throws DOMException;

    public void removeAttribute(String name) throws DOMException;

    @Pure
    @Nullable
    public Attr getAttributeNode(String name);

    @Nullable
    public Attr setAttributeNode(Attr newAttr) throws DOMException;

    public Attr removeAttributeNode(Attr oldAttr) throws DOMException;

    @Pure
    public NodeList getElementsByTagName(String name);

    @Pure
    public String getAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    public void setAttributeNS(@Nullable String namespaceURI, String qualifiedName, String value) throws DOMException;

    public void removeAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Pure
    @Nullable
    public Attr getAttributeNodeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Nullable
    public Attr setAttributeNodeNS(Attr newAttr) throws DOMException;

    @Pure
    public NodeList getElementsByTagNameNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Pure
    public boolean hasAttribute(String name);

    @Pure
    public boolean hasAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Pure
    public TypeInfo getSchemaTypeInfo();

    public void setIdAttribute(String name, boolean isId) throws DOMException;

    public void setIdAttributeNS(@Nullable String namespaceURI, String localName, boolean isId) throws DOMException;

    public void setIdAttributeNode(Attr idAttr, boolean isId) throws DOMException;
}

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
@AnnotatedFor({ "nullness" })
    @Positive
public interface Element extends Node {

    @Positive
    @Pure
    @Positive
    public String getTagName();

    @Positive
    @Pure
    @Positive
    public String getAttribute(String name);

    @Positive
    public void setAttribute(String name, String value) throws DOMException;

    @Positive
    public void removeAttribute(String name) throws DOMException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Attr getAttributeNode(String name);

    @Positive
    @Nullable
    @Positive
    public Attr setAttributeNode(Attr newAttr) throws DOMException;

    @Positive
    public Attr removeAttributeNode(Attr oldAttr) throws DOMException;

    @Positive
    @Pure
    @Positive
    public NodeList getElementsByTagName(String name);

    @Positive
    @Pure
    @Positive
    public String getAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Positive
    public void setAttributeNS(@Nullable String namespaceURI, String qualifiedName, String value) throws DOMException;

    @Positive
    public void removeAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Attr getAttributeNodeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Positive
    @Nullable
    @Positive
    public Attr setAttributeNodeNS(Attr newAttr) throws DOMException;

    @Positive
    @Pure
    @Positive
    public NodeList getElementsByTagNameNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Positive
    @Pure
    @Positive
    public boolean hasAttribute(String name);

    @Positive
    @Pure
    @Positive
    public boolean hasAttributeNS(@Nullable String namespaceURI, String localName) throws DOMException;

    @Positive
    @Pure
    @Positive
    public TypeInfo getSchemaTypeInfo();

    @Positive
    public void setIdAttribute(String name, boolean isId) throws DOMException;

    @Positive
    public void setIdAttributeNS(@Nullable String namespaceURI, String localName, boolean isId) throws DOMException;

    @Positive
    public void setIdAttributeNode(Attr idAttr, boolean isId) throws DOMException;
    @Positive
}

// CFWR semantic augmentation - variant 1

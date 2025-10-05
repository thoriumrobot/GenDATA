/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package javax.xml.stream;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import javax.xml.namespace.NamespaceContext;
    @Positive
import javax.xml.namespace.QName;

    @Positive
public interface XMLStreamReader extends XMLStreamConstants {

    @Positive
    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    @Positive
    public int next() throws XMLStreamException;

    @Positive
    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    @Positive
    public String getElementText() throws XMLStreamException;

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public int nextTag() throws XMLStreamException;

    @Positive
    @Pure
    @Positive
    public boolean hasNext() throws XMLStreamException;

    @Positive
    public void close() throws XMLStreamException;

    @Positive
    public String getNamespaceURI(String prefix);

    @Positive
    public boolean isStartElement();

    @Positive
    public boolean isEndElement();

    @Positive
    public boolean isCharacters();

    @Positive
    public boolean isWhiteSpace();

    @Positive
    public String getAttributeValue(String namespaceURI, String localName);

    @Positive
    public int getAttributeCount();

    @Positive
    public QName getAttributeName(int index);

    @Positive
    public String getAttributeNamespace(int index);

    @Positive
    public String getAttributeLocalName(int index);

    @Positive
    public String getAttributePrefix(int index);

    @Positive
    public String getAttributeType(int index);

    @Positive
    public String getAttributeValue(int index);

    @Positive
    public boolean isAttributeSpecified(int index);

    @Positive
    public int getNamespaceCount();

    @Positive
    public String getNamespacePrefix(int index);

    @Positive
    public String getNamespaceURI(int index);

    @Positive
    public NamespaceContext getNamespaceContext();

    @Positive
    public int getEventType();

    @Positive
    public String getText();

    @Positive
    public char[] getTextCharacters();

    @Positive
    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    @Positive
    public int getTextStart();

    @Positive
    public int getTextLength();

    @Positive
    public String getEncoding();

    @Positive
    public boolean hasText();

    @Positive
    public Location getLocation();

    @Positive
    public QName getName();

    @Positive
    public String getLocalName();

    @Positive
    public boolean hasName();

    @Positive
    public String getNamespaceURI();

    @Positive
    public String getPrefix();

    @Positive
    public String getVersion();

    @Positive
    public boolean isStandalone();

    @Positive
    public boolean standaloneSet();

    @Positive
    public String getCharacterEncodingScheme();

    @Positive
    public String getPITarget();

    @Positive
    public String getPIData();
    @Positive
}

/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

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
import java.util.*;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;

    @Positive
public class AttributedString {

    @Positive
    public AttributedString(String text) {
    @Positive
    }

    @Positive
    public AttributedString(String text, Map<? extends Attribute, ?> attributes) {
    @Positive
    }

    @Positive
    public AttributedString(AttributedCharacterIterator text) {
    @Positive
    }

    @Positive
    public AttributedString(AttributedCharacterIterator text, int beginIndex, int endIndex) {
    @Positive
    }

    @Positive
    public AttributedString(AttributedCharacterIterator text, int beginIndex, int endIndex, Attribute[] attributes) {
    @Positive
    }

    @Positive
    public void addAttribute(Attribute attribute, Object value);

    @Positive
    public void addAttribute(Attribute attribute, Object value, int beginIndex, int endIndex);

    @Positive
    public void addAttributes(Map<? extends Attribute, ?> attributes, int beginIndex, int endIndex);

    @Positive
    public AttributedCharacterIterator getIterator();

    @Positive
    public AttributedCharacterIterator getIterator(Attribute[] attributes);

    @Positive
    public AttributedCharacterIterator getIterator(Attribute[] attributes, int beginIndex, int endIndex);

    @Positive
    int length();

    @Positive
    private final class AttributedStringIterator implements AttributedCharacterIterator {

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();

    @Positive
        public Object clone();

    @Positive
        public char first();

    @Positive
        public char last();

    @Positive
        public char current();

    @Positive
        public char next();

    @Positive
        public char previous();

    @Positive
        public char setIndex(int position);

    @Positive
        public int getBeginIndex();

    @Positive
        public int getEndIndex();

    @Positive
        public int getIndex();

    @Positive
        public int getRunStart();

    @Positive
        public int getRunStart(Attribute attribute);

    @Positive
        public int getRunStart(Set<? extends Attribute> attributes);

    @Positive
        public int getRunLimit();

    @Positive
        public int getRunLimit(Attribute attribute);

    @Positive
        public int getRunLimit(Set<? extends Attribute> attributes);

    @Positive
        public Map<Attribute, Object> getAttributes();

    @Positive
        public Set<Attribute> getAllAttributeKeys();

    @Positive
        public Object getAttribute(Attribute attribute);
    @Positive
    }

    @Positive
    private final class AttributeMap extends AbstractMap<Attribute, Object> {

    @Positive
        public Set<Map.Entry<Attribute, Object>> entrySet();

    @Positive
        public Object get(Object key);
    @Positive
    }
    @Positive
}

    @Positive
class AttributeEntry implements Map.Entry<Attribute, Object> {

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public Attribute getKey();

    @Positive
    public Object getValue();

    @Positive
    public Object setValue(Object newValue);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}

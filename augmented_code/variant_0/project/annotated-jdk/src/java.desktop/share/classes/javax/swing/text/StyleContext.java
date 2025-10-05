/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.text;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.FontMetrics;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.NotSerializableException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Vector;
    @Positive
import java.util.WeakHashMap;
    @Positive
import javax.swing.SwingUtilities;
    @Positive
import javax.swing.event.ChangeEvent;
    @Positive
import javax.swing.event.ChangeListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import sun.font.FontUtilities;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class StyleContext implements Serializable, AbstractDocument.AttributeContext {

    @Positive
    public static final StyleContext getDefaultStyleContext();

    @Positive
    public StyleContext() {
    @Positive
    }

    @Positive
    public Style addStyle(String nm, Style parent);

    @Positive
    public void removeStyle(String nm);

    @Positive
    public Style getStyle(String nm);

    @Positive
    public Enumeration<?> getStyleNames();

    @Positive
    public void addChangeListener(ChangeListener l);

    @Positive
    public void removeChangeListener(ChangeListener l);

    @Positive
    public ChangeListener[] getChangeListeners();

    @Positive
    public Font getFont(AttributeSet attr);

    @Positive
    public Color getForeground(AttributeSet attr);

    @Positive
    public Color getBackground(AttributeSet attr);

    @Positive
    public Font getFont(String family, int style, int size);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public FontMetrics getFontMetrics(Font f);

    @Positive
    public synchronized AttributeSet addAttribute(AttributeSet old, Object name, Object value);

    @Positive
    public synchronized AttributeSet addAttributes(AttributeSet old, AttributeSet attr);

    @Positive
    public synchronized AttributeSet removeAttribute(AttributeSet old, Object name);

    @Positive
    public synchronized AttributeSet removeAttributes(AttributeSet old, Enumeration<?> names);

    @Positive
    public synchronized AttributeSet removeAttributes(AttributeSet old, AttributeSet attrs);

    @Positive
    public AttributeSet getEmptySet();

    @Positive
    public void reclaim(AttributeSet a);

    @Positive
    protected int getCompressionThreshold();

    @Positive
    protected SmallAttributeSet createSmallAttributeSet(AttributeSet a);

    @Positive
    protected MutableAttributeSet createLargeAttributeSet(AttributeSet a);

    @Positive
    synchronized void removeUnusedSets();

    @Positive
    AttributeSet getImmutableUniqueSet();

    @Positive
    MutableAttributeSet getMutableAttributeSet(AttributeSet a);

    @Positive
    public String toString();

    @Positive
    public void writeAttributes(ObjectOutputStream out, AttributeSet a) throws IOException;

    @Positive
    public void readAttributes(ObjectInputStream in, MutableAttributeSet a) throws ClassNotFoundException, IOException;

    @Positive
    public static void writeAttributeSet(ObjectOutputStream out, AttributeSet a) throws IOException;

    @Positive
    public static void readAttributeSet(ObjectInputStream in, MutableAttributeSet a) throws ClassNotFoundException, IOException;

    @Positive
    public static void registerStaticAttributeKey(Object key);

    @Positive
    public static Object getStaticAttribute(Object key);

    @Positive
    public static Object getStaticAttributeKey(Object key);

    @Positive
    @Interned
    @Positive
    public static final String DEFAULT_STYLE;

    @Positive
    public class SmallAttributeSet implements AttributeSet {

    @Positive
        public SmallAttributeSet(Object[] attributes) {
    @Positive
        }

    @Positive
        public SmallAttributeSet(AttributeSet attrs) {
    @Positive
        }

    @Positive
        Object getLocalAttribute(Object nm);

    @Positive
        public String toString();

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);

    @Positive
        public Object clone();

    @Positive
        public int getAttributeCount();

    @Positive
        public boolean isDefined(Object key);

    @Positive
        public boolean isEqual(AttributeSet attr);

    @Positive
        public AttributeSet copyAttributes();

    @Positive
        public Object getAttribute(Object key);

    @Positive
        public Enumeration<?> getAttributeNames();

    @Positive
        public boolean containsAttribute(Object name, Object value);

    @Positive
        public boolean containsAttributes(AttributeSet attrs);

    @Positive
        public AttributeSet getResolveParent();
    @Positive
    }

    @Positive
    class KeyEnumeration implements Enumeration<Object> {

    @Positive
        public boolean hasMoreElements();

    @Positive
        public Object nextElement();
    @Positive
    }

    @Positive
    class KeyBuilder {

    @Positive
        public void initialize(AttributeSet a);

    @Positive
        public Object[] createTable();

    @Positive
        int getCount();

    @Positive
        public void addAttribute(Object key, Object value);

    @Positive
        public void addAttributes(AttributeSet attr);

    @Positive
        public void removeAttribute(Object key);

    @Positive
        public void removeAttributes(Enumeration<?> names);

    @Positive
        public void removeAttributes(AttributeSet attr);
    @Positive
    }

    @Positive
    static class FontKey {

    @Positive
        public FontKey(String family, int style, int size) {
    @Positive
        }

    @Positive
        public void setValue(String family, int style, int size);

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public class NamedStyle implements Style, Serializable {

    @Positive
        public NamedStyle(String name, Style parent) {
    @Positive
        }

    @Positive
        public NamedStyle(Style parent) {
    @Positive
        }

    @Positive
        public NamedStyle() {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        public String getName();

    @Positive
        public void setName(String name);

    @Positive
        public void addChangeListener(ChangeListener l);

    @Positive
        public void removeChangeListener(ChangeListener l);

    @Positive
        public ChangeListener[] getChangeListeners();

    @Positive
        protected void fireStateChanged();

    @Positive
        public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
        public int getAttributeCount();

    @Positive
        public boolean isDefined(Object attrName);

    @Positive
        public boolean isEqual(AttributeSet attr);

    @Positive
        public AttributeSet copyAttributes();

    @Positive
        public Object getAttribute(Object attrName);

    @Positive
        public Enumeration<?> getAttributeNames();

    @Positive
        public boolean containsAttribute(Object name, Object value);

    @Positive
        public boolean containsAttributes(AttributeSet attrs);

    @Positive
        public AttributeSet getResolveParent();

    @Positive
        public void addAttribute(Object name, Object value);

    @Positive
        public void addAttributes(AttributeSet attr);

    @Positive
        public void removeAttribute(Object name);

    @Positive
        public void removeAttributes(Enumeration<?> names);

    @Positive
        public void removeAttributes(AttributeSet attrs);

    @Positive
        public void setResolveParent(AttributeSet parent);

    @Positive
        protected EventListenerList listenerList;

    @Positive
        protected transient ChangeEvent changeEvent;
    @Positive
    }
    @Positive
}

/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.text;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Toolkit;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.EventListener;
import java.util.Hashtable;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Vector;
import java.util.WeakHashMap;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.EventListenerList;
import sun.font.FontUtilities;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class StyleContext implements Serializable, AbstractDocument.AttributeContext {

    public static final StyleContext getDefaultStyleContext();

    public StyleContext() {
    }

    public Style addStyle(String nm, Style parent);

    public void removeStyle(String nm);

    public Style getStyle(String nm);

    public Enumeration<?> getStyleNames();

    public void addChangeListener(ChangeListener l);

    public void removeChangeListener(ChangeListener l);

    public ChangeListener[] getChangeListeners();

    public Font getFont(AttributeSet attr);

    public Color getForeground(AttributeSet attr);

    public Color getBackground(AttributeSet attr);

    public Font getFont(String family, int style, int size);

    @SuppressWarnings("deprecation")
    public FontMetrics getFontMetrics(Font f);

    public synchronized AttributeSet addAttribute(AttributeSet old, Object name, Object value);

    public synchronized AttributeSet addAttributes(AttributeSet old, AttributeSet attr);

    public synchronized AttributeSet removeAttribute(AttributeSet old, Object name);

    public synchronized AttributeSet removeAttributes(AttributeSet old, Enumeration<?> names);

    public synchronized AttributeSet removeAttributes(AttributeSet old, AttributeSet attrs);

    public AttributeSet getEmptySet();

    public void reclaim(AttributeSet a);

    protected int getCompressionThreshold();

    protected SmallAttributeSet createSmallAttributeSet(AttributeSet a);

    protected MutableAttributeSet createLargeAttributeSet(AttributeSet a);

    synchronized void removeUnusedSets();

    AttributeSet getImmutableUniqueSet();

    MutableAttributeSet getMutableAttributeSet(AttributeSet a);

    public String toString();

    public void writeAttributes(ObjectOutputStream out, AttributeSet a) throws IOException;

    public void readAttributes(ObjectInputStream in, MutableAttributeSet a) throws ClassNotFoundException, IOException;

    public static void writeAttributeSet(ObjectOutputStream out, AttributeSet a) throws IOException;

    public static void readAttributeSet(ObjectInputStream in, MutableAttributeSet a) throws ClassNotFoundException, IOException;

    public static void registerStaticAttributeKey(Object key);

    public static Object getStaticAttribute(Object key);

    public static Object getStaticAttributeKey(Object key);

    @Interned
    public static final String DEFAULT_STYLE;

    public class SmallAttributeSet implements AttributeSet {

        public SmallAttributeSet(Object[] attributes) {
        }

        public SmallAttributeSet(AttributeSet attrs) {
        }

        Object getLocalAttribute(Object nm);

        public String toString();

        public int hashCode();

        public boolean equals(Object obj);

        public Object clone();

        public int getAttributeCount();

        public boolean isDefined(Object key);

        public boolean isEqual(AttributeSet attr);

        public AttributeSet copyAttributes();

        public Object getAttribute(Object key);

        public Enumeration<?> getAttributeNames();

        public boolean containsAttribute(Object name, Object value);

        public boolean containsAttributes(AttributeSet attrs);

        public AttributeSet getResolveParent();
    }

    class KeyEnumeration implements Enumeration<Object> {

        public boolean hasMoreElements();

        public Object nextElement();
    }

    class KeyBuilder {

        public void initialize(AttributeSet a);

        public Object[] createTable();

        int getCount();

        public void addAttribute(Object key, Object value);

        public void addAttributes(AttributeSet attr);

        public void removeAttribute(Object key);

        public void removeAttributes(Enumeration<?> names);

        public void removeAttributes(AttributeSet attr);
    }

    static class FontKey {

        public FontKey(String family, int style, int size) {
        }

        public void setValue(String family, int style, int size);

        public int hashCode();

        public boolean equals(Object obj);
    }

    @SuppressWarnings("serial")
    public class NamedStyle implements Style, Serializable {

        public NamedStyle(String name, Style parent) {
        }

        public NamedStyle(Style parent) {
        }

        public NamedStyle() {
        }

        public String toString();

        public String getName();

        public void setName(String name);

        public void addChangeListener(ChangeListener l);

        public void removeChangeListener(ChangeListener l);

        public ChangeListener[] getChangeListeners();

        protected void fireStateChanged();

        public <T extends EventListener> T[] getListeners(Class<T> listenerType);

        public int getAttributeCount();

        public boolean isDefined(Object attrName);

        public boolean isEqual(AttributeSet attr);

        public AttributeSet copyAttributes();

        public Object getAttribute(Object attrName);

        public Enumeration<?> getAttributeNames();

        public boolean containsAttribute(Object name, Object value);

        public boolean containsAttributes(AttributeSet attrs);

        public AttributeSet getResolveParent();

        public void addAttribute(Object name, Object value);

        public void addAttributes(AttributeSet attr);

        public void removeAttribute(Object name);

        public void removeAttributes(Enumeration<?> names);

        public void removeAttributes(AttributeSet attrs);

        public void setResolveParent(AttributeSet parent);

        protected EventListenerList listenerList;

        protected transient ChangeEvent changeEvent;
    }
}

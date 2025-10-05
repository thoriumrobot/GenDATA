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
package javax.swing;

    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.swing.border.*;
    @Positive
import javax.swing.event.SwingPropertyChangeSupport;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.StringWriter;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.reflect.*;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Vector;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Dimension;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import sun.reflect.misc.MethodUtil;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.swing.SwingAccessor;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class UIDefaults extends Hashtable<Object, Object> {

    @Positive
    public UIDefaults() {
    @Positive
    }

    @Positive
    public UIDefaults(int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public UIDefaults(Object[] keyValueList) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public Object get(Object key);

    @Positive
    @Nullable
    @Positive
    public Object get(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Object put(Object key, @Nullable Object value);

    @Positive
    public void putDefaults(@Nullable Object[] keyValueList);

    @Positive
    @Nullable
    @Positive
    public Font getFont(Object key);

    @Positive
    @Nullable
    @Positive
    public Font getFont(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Color getColor(Object key);

    @Positive
    @Nullable
    @Positive
    public Color getColor(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Icon getIcon(Object key);

    @Positive
    @Nullable
    @Positive
    public Icon getIcon(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Border getBorder(Object key);

    @Positive
    @Nullable
    @Positive
    public Border getBorder(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public String getString(Object key);

    @Positive
    @Nullable
    @Positive
    public String getString(Object key, @Nullable Locale l);

    @Positive
    public int getInt(Object key);

    @Positive
    public int getInt(Object key, @Nullable Locale l);

    @Positive
    public boolean getBoolean(Object key);

    @Positive
    public boolean getBoolean(Object key, Locale l);

    @Positive
    @Nullable
    @Positive
    public Insets getInsets(Object key);

    @Positive
    @Nullable
    @Positive
    public Insets getInsets(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Dimension getDimension(Object key);

    @Positive
    @Nullable
    @Positive
    public Dimension getDimension(Object key, @Nullable Locale l);

    @Positive
    @Nullable
    @Positive
    public Class<? extends ComponentUI> getUIClass(String uiClassID, @Nullable ClassLoader uiClassLoader);

    @Positive
    @Nullable
    @Positive
    public Class<? extends ComponentUI> getUIClass(String uiClassID);

    @Positive
    protected void getUIError(String msg);

    @Positive
    public ComponentUI getUI(JComponent target);

    @Positive
    public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public synchronized PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
    protected void firePropertyChange(String propertyName, Object oldValue, Object newValue);

    @Positive
    public synchronized void addResourceBundle(final String bundleName);

    @Positive
    public synchronized void removeResourceBundle(String bundleName);

    @Positive
    public void setDefaultLocale(Locale l);

    @Positive
    public Locale getDefaultLocale();

    @Positive
    public interface LazyValue {

    @Positive
        Object createValue(UIDefaults table);
    @Positive
    }

    @Positive
    public interface ActiveValue {

    @Positive
        Object createValue(UIDefaults table);
    @Positive
    }

    @Positive
    public static class ProxyLazyValue implements LazyValue {

    @Positive
        public ProxyLazyValue(String c) {
    @Positive
        }

    @Positive
        public ProxyLazyValue(String c, String m) {
    @Positive
        }

    @Positive
        public ProxyLazyValue(String c, Object[] o) {
    @Positive
        }

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public ProxyLazyValue(String c, String m, Object[] o) {
    @Positive
        }

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public Object createValue(final UIDefaults table);
    @Positive
    }

    @Positive
    public static class LazyInputMap implements LazyValue {

    @Positive
        public LazyInputMap(Object[] bindings) {
    @Positive
        }

    @Positive
        public Object createValue(UIDefaults table);
    @Positive
    }

    @Positive
    private static class TextAndMnemonicHashMap extends HashMap<String, Object> {

    @Positive
        @Override
    @Positive
        public Object get(Object key);

    @Positive
        String composeKey(String key, int reduce, String sufix);

    @Positive
        String getTextFromProperty(String text);

    @Positive
        String getMnemonicFromProperty(String text);

    @Positive
        String getIndexFromProperty(String text);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1

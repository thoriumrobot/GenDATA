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
package javax.swing;

import javax.swing.plaf.ComponentUI;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.swing.border.*;
import javax.swing.event.SwingPropertyChangeSupport;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.lang.reflect.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.ResourceBundle;
import java.util.Locale;
import java.util.Vector;
import java.util.MissingResourceException;
import java.awt.Font;
import java.awt.Color;
import java.awt.Insets;
import java.awt.Dimension;
import java.beans.PropertyChangeListener;
import java.security.AccessController;
import java.security.AccessControlContext;
import java.security.PrivilegedAction;
import sun.reflect.misc.MethodUtil;
import sun.reflect.misc.ReflectUtil;
import sun.swing.SwingAccessor;
import sun.swing.SwingUtilities2;

@AnnotatedFor({ "nullness" })
@SuppressWarnings("serial")
public class UIDefaults extends Hashtable<Object, Object> {

    public UIDefaults() {
    }

    public UIDefaults(int initialCapacity, float loadFactor) {
    }

    public UIDefaults(Object[] keyValueList) {
    }

    @Nullable
    public Object get(Object key);

    @Nullable
    public Object get(Object key, @Nullable Locale l);

    @Nullable
    public Object put(Object key, @Nullable Object value);

    public void putDefaults(@Nullable Object[] keyValueList);

    @Nullable
    public Font getFont(Object key);

    @Nullable
    public Font getFont(Object key, @Nullable Locale l);

    @Nullable
    public Color getColor(Object key);

    @Nullable
    public Color getColor(Object key, @Nullable Locale l);

    @Nullable
    public Icon getIcon(Object key);

    @Nullable
    public Icon getIcon(Object key, @Nullable Locale l);

    @Nullable
    public Border getBorder(Object key);

    @Nullable
    public Border getBorder(Object key, @Nullable Locale l);

    @Nullable
    public String getString(Object key);

    @Nullable
    public String getString(Object key, @Nullable Locale l);

    public int getInt(Object key);

    public int getInt(Object key, @Nullable Locale l);

    public boolean getBoolean(Object key);

    public boolean getBoolean(Object key, Locale l);

    @Nullable
    public Insets getInsets(Object key);

    @Nullable
    public Insets getInsets(Object key, @Nullable Locale l);

    @Nullable
    public Dimension getDimension(Object key);

    @Nullable
    public Dimension getDimension(Object key, @Nullable Locale l);

    @Nullable
    public Class<? extends ComponentUI> getUIClass(String uiClassID, @Nullable ClassLoader uiClassLoader);

    @Nullable
    public Class<? extends ComponentUI> getUIClass(String uiClassID);

    protected void getUIError(String msg);

    public ComponentUI getUI(JComponent target);

    public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

    public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

    public synchronized PropertyChangeListener[] getPropertyChangeListeners();

    protected void firePropertyChange(String propertyName, Object oldValue, Object newValue);

    public synchronized void addResourceBundle(final String bundleName);

    public synchronized void removeResourceBundle(String bundleName);

    public void setDefaultLocale(Locale l);

    public Locale getDefaultLocale();

    public interface LazyValue {

        Object createValue(UIDefaults table);
    }

    public interface ActiveValue {

        Object createValue(UIDefaults table);
    }

    public static class ProxyLazyValue implements LazyValue {

        public ProxyLazyValue(String c) {
        }

        public ProxyLazyValue(String c, String m) {
        }

        public ProxyLazyValue(String c, Object[] o) {
        }

        @SuppressWarnings("removal")
        public ProxyLazyValue(String c, String m, Object[] o) {
        }

        @SuppressWarnings("removal")
        public Object createValue(final UIDefaults table);
    }

    public static class LazyInputMap implements LazyValue {

        public LazyInputMap(Object[] bindings) {
        }

        public Object createValue(UIDefaults table);
    }

    private static class TextAndMnemonicHashMap extends HashMap<String, Object> {

        @Override
        public Object get(Object key);

        String composeKey(String key, int reduce, String sufix);

        String getTextFromProperty(String text);

        String getMnemonicFromProperty(String text);

        String getIndexFromProperty(String text);
    }
}

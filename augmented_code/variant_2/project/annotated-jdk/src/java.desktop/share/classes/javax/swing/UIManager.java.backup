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

import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.guieffect.qual.UIType;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Component;
import java.awt.Font;
import java.awt.Color;
import java.awt.Insets;
import java.awt.Dimension;
import java.awt.KeyboardFocusManager;
import java.awt.KeyEventPostProcessor;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.security.AccessController;
import javax.swing.plaf.ComponentUI;
import javax.swing.border.Border;
import javax.swing.event.SwingPropertyChangeSupport;
import java.beans.PropertyChangeListener;
import java.io.Serializable;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.Locale;
import sun.awt.SunToolkit;
import sun.awt.OSInfo;
import sun.security.action.GetPropertyAction;
import sun.swing.SwingUtilities2;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Objects;
import sun.awt.AppContext;
import sun.awt.AWTAccessor;

@AnnotatedFor({ "guieffect", "nullness" })
@UIType
@SuppressWarnings("serial")
public class UIManager implements Serializable {

    private static class LAFState {

        UIDefaults getLookAndFeelDefaults();

        void setLookAndFeelDefaults(UIDefaults x);

        UIDefaults getSystemDefaults();

        void setSystemDefaults(UIDefaults x);

        public synchronized SwingPropertyChangeSupport getPropertyChangeSupport(boolean create);
    }

    public UIManager() {
    }

    public static class LookAndFeelInfo {

        public LookAndFeelInfo(String name, String className) {
        }

        public String getName();

        public String getClassName();

        public String toString();
    }

    public static LookAndFeelInfo[] getInstalledLookAndFeels();

    public static void setInstalledLookAndFeels(LookAndFeelInfo[] infos) throws SecurityException;

    public static void installLookAndFeel(LookAndFeelInfo info);

    public static void installLookAndFeel(String name, String className);

    public static LookAndFeel getLookAndFeel();

    @SuppressWarnings("deprecation")
    public static LookAndFeel createLookAndFeel(String name) throws UnsupportedLookAndFeelException;

    @SafeEffect
    public static void setLookAndFeel(LookAndFeel newLookAndFeel) throws UnsupportedLookAndFeelException;

    @SafeEffect
    @SuppressWarnings("deprecation")
    public static void setLookAndFeel(String className) throws ClassNotFoundException, InstantiationException, IllegalAccessException, UnsupportedLookAndFeelException;

    public static String getSystemLookAndFeelClassName();

    public static String getCrossPlatformLookAndFeelClassName();

    public static UIDefaults getDefaults();

    public static Font getFont(Object key);

    public static Font getFont(Object key, Locale l);

    public static Color getColor(Object key);

    public static Color getColor(Object key, Locale l);

    public static Icon getIcon(Object key);

    public static Icon getIcon(Object key, Locale l);

    public static Border getBorder(Object key);

    public static Border getBorder(Object key, Locale l);

    public static String getString(Object key);

    public static String getString(Object key, Locale l);

    static String getString(Object key, Component c);

    public static int getInt(Object key);

    public static int getInt(Object key, Locale l);

    public static boolean getBoolean(Object key);

    public static boolean getBoolean(Object key, Locale l);

    public static Insets getInsets(Object key);

    public static Insets getInsets(Object key, Locale l);

    public static Dimension getDimension(Object key);

    public static Dimension getDimension(Object key, Locale l);

    @Nullable
    public static Object get(Object key);

    @Nullable
    public static Object get(Object key, Locale l);

    @Nullable
    public static Object put(Object key, @Nullable Object value);

    public static ComponentUI getUI(JComponent target);

    public static UIDefaults getLookAndFeelDefaults();

    public static void addAuxiliaryLookAndFeel(LookAndFeel laf);

    public static boolean removeAuxiliaryLookAndFeel(LookAndFeel laf);

    public static LookAndFeel[] getAuxiliaryLookAndFeels();

    public static void addPropertyChangeListener(PropertyChangeListener listener);

    public static void removePropertyChangeListener(PropertyChangeListener listener);

    public static PropertyChangeListener[] getPropertyChangeListeners();
}

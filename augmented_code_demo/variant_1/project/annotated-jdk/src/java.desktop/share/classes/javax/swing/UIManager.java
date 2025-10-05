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
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.KeyboardFocusManager;
    @Positive
import java.awt.KeyEventPostProcessor;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.security.AccessController;
    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import javax.swing.border.Border;
    @Positive
import javax.swing.event.SwingPropertyChangeSupport;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.Serializable;
    @Positive
import java.io.File;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Properties;
    @Positive
import java.util.StringTokenizer;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Locale;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.OSInfo;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Objects;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.AWTAccessor;

    @Positive
@AnnotatedFor({ "guieffect", "nullness" })
    @Positive
@UIType
    @Positive
@SuppressWarnings("serial")
    @Positive
public class UIManager implements Serializable {

    @Positive
    private static class LAFState {

    @Positive
        UIDefaults getLookAndFeelDefaults();

    @Positive
        void setLookAndFeelDefaults(UIDefaults x);

    @Positive
        UIDefaults getSystemDefaults();

    @Positive
        void setSystemDefaults(UIDefaults x);

    @Positive
        public synchronized SwingPropertyChangeSupport getPropertyChangeSupport(boolean create);
    @Positive
    }

    @Positive
    public UIManager() {
    @Positive
    }

    @Positive
    public static class LookAndFeelInfo {

    @Positive
        public LookAndFeelInfo(String name, String className) {
    @Positive
        }

    @Positive
        public String getName();

    @Positive
        public String getClassName();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static LookAndFeelInfo[] getInstalledLookAndFeels();

    @Positive
    public static void setInstalledLookAndFeels(LookAndFeelInfo[] infos) throws SecurityException;

    @Positive
    public static void installLookAndFeel(LookAndFeelInfo info);

    @Positive
    public static void installLookAndFeel(String name, String className);

    @Positive
    public static LookAndFeel getLookAndFeel();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static LookAndFeel createLookAndFeel(String name) throws UnsupportedLookAndFeelException;

    @Positive
    @SafeEffect
    @Positive
    public static void setLookAndFeel(LookAndFeel newLookAndFeel) throws UnsupportedLookAndFeelException;

    @Positive
    @SafeEffect
    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static void setLookAndFeel(String className) throws ClassNotFoundException, InstantiationException, IllegalAccessException, UnsupportedLookAndFeelException;

    @Positive
    public static String getSystemLookAndFeelClassName();

    @Positive
    public static String getCrossPlatformLookAndFeelClassName();

    @Positive
    public static UIDefaults getDefaults();

    @Positive
    public static Font getFont(Object key);

    @Positive
    public static Font getFont(Object key, Locale l);

    @Positive
    public static Color getColor(Object key);

    @Positive
    public static Color getColor(Object key, Locale l);

    @Positive
    public static Icon getIcon(Object key);

    @Positive
    public static Icon getIcon(Object key, Locale l);

    @Positive
    public static Border getBorder(Object key);

    @Positive
    public static Border getBorder(Object key, Locale l);

    @Positive
    public static String getString(Object key);

    @Positive
    public static String getString(Object key, Locale l);

    @Positive
    static String getString(Object key, Component c);

    @Positive
    public static int getInt(Object key);

    @Positive
    public static int getInt(Object key, Locale l);

    @Positive
    public static boolean getBoolean(Object key);

    @Positive
    public static boolean getBoolean(Object key, Locale l);

    @Positive
    public static Insets getInsets(Object key);

    @Positive
    public static Insets getInsets(Object key, Locale l);

    @Positive
    public static Dimension getDimension(Object key);

    @Positive
    public static Dimension getDimension(Object key, Locale l);

    @Positive
    @Nullable
    @Positive
    public static Object get(Object key);

    @Positive
    @Nullable
    @Positive
    public static Object get(Object key, Locale l);

    @Positive
    @Nullable
    @Positive
    public static Object put(Object key, @Nullable Object value);

    @Positive
    public static ComponentUI getUI(JComponent target);

    @Positive
    public static UIDefaults getLookAndFeelDefaults();

    @Positive
    public static void addAuxiliaryLookAndFeel(LookAndFeel laf);

    @Positive
    public static boolean removeAuxiliaryLookAndFeel(LookAndFeel laf);

    @Positive
    public static LookAndFeel[] getAuxiliaryLookAndFeels();

    @Positive
    public static void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public static void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public static PropertyChangeListener[] getPropertyChangeListeners();
    @Positive
}

// CFWR semantic augmentation - variant 1

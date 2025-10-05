/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.datatransfer.Clipboard;
import java.awt.dnd.DragGestureListener;
import java.awt.dnd.DragGestureRecognizer;
import java.awt.dnd.DragSource;
import java.awt.event.AWTEventListener;
import java.awt.event.AWTEventListenerProxy;
import java.awt.event.ActionEvent;
import java.awt.event.AdjustmentEvent;
import java.awt.event.ComponentEvent;
import java.awt.event.ContainerEvent;
import java.awt.event.FocusEvent;
import java.awt.event.HierarchyEvent;
import java.awt.event.InputEvent;
import java.awt.event.InputMethodEvent;
import java.awt.event.InvocationEvent;
import java.awt.event.ItemEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.PaintEvent;
import java.awt.event.TextEvent;
import java.awt.event.WindowEvent;
import java.awt.im.InputMethodHighlight;
import java.awt.image.ColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.ImageProducer;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EventListener;
import java.util.HashMap;
import java.util.Map;
import java.util.MissingResourceException;
import java.util.Properties;
import java.util.ResourceBundle;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.stream.Collectors;
import javax.accessibility.AccessibilityProvider;
import sun.awt.AWTAccessor;
import sun.awt.AWTPermissions;
import sun.awt.AppContext;
import sun.awt.HeadlessToolkit;
import sun.awt.PeerEvent;
import sun.awt.PlatformGraphicsInfo;
import sun.awt.SunToolkit;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class Toolkit {

    protected Toolkit() {
    }

    protected void loadSystemColors(int[] systemColors) throws HeadlessException;

    public void setDynamicLayout(final boolean dynamic) throws HeadlessException;

    protected boolean isDynamicLayoutSet() throws HeadlessException;

    public boolean isDynamicLayoutActive() throws HeadlessException;

    public abstract Dimension getScreenSize() throws HeadlessException;

    public abstract int getScreenResolution() throws HeadlessException;

    public Insets getScreenInsets(GraphicsConfiguration gc) throws HeadlessException;

    public abstract ColorModel getColorModel() throws HeadlessException;

    @Deprecated
    public abstract String[] getFontList();

    @Deprecated
    public abstract FontMetrics getFontMetrics(Font font);

    public abstract void sync();

    public static synchronized Toolkit getDefaultToolkit();

    public abstract Image getImage(String filename);

    public abstract Image getImage(URL url);

    public abstract Image createImage(String filename);

    public abstract Image createImage(URL url);

    public abstract boolean prepareImage(Image image, int width, int height, ImageObserver observer);

    public abstract int checkImage(Image image, int width, int height, ImageObserver observer);

    public abstract Image createImage(ImageProducer producer);

    public Image createImage(byte[] imagedata);

    public abstract Image createImage(byte[] imagedata, int imageoffset, int imagelength);

    public abstract PrintJob getPrintJob(Frame frame, String jobtitle, Properties props);

    public PrintJob getPrintJob(Frame frame, String jobtitle, JobAttributes jobAttributes, PageAttributes pageAttributes);

    public abstract void beep();

    public abstract Clipboard getSystemClipboard() throws HeadlessException;

    public Clipboard getSystemSelection() throws HeadlessException;

    @Deprecated()
    public int getMenuShortcutKeyMask() throws HeadlessException;

    public int getMenuShortcutKeyMaskEx() throws HeadlessException;

    public boolean getLockingKeyState(int keyCode) throws UnsupportedOperationException;

    public void setLockingKeyState(int keyCode, boolean on) throws UnsupportedOperationException;

    protected static Container getNativeContainer(Component c);

    public Cursor createCustomCursor(Image cursor, Point hotSpot, String name) throws IndexOutOfBoundsException, HeadlessException;

    public Dimension getBestCursorSize(int preferredWidth, int preferredHeight) throws HeadlessException;

    public int getMaximumCursorColors() throws HeadlessException;

    public boolean isFrameStateSupported(int state) throws HeadlessException;

    @SuppressWarnings("removal")
    static void loadLibraries();

    public static String getProperty(String key, String defaultValue);

    public final EventQueue getSystemEventQueue();

    protected abstract EventQueue getSystemEventQueueImpl();

    static EventQueue getEventQueue();

    public <T extends DragGestureRecognizer> T createDragGestureRecognizer(Class<T> abstractRecognizerClass, DragSource ds, Component c, int srcActions, DragGestureListener dgl);

    public final synchronized Object getDesktopProperty(String propertyName);

    protected final void setDesktopProperty(String name, Object newValue);

    protected Object lazilyLoadDesktopProperty(String name);

    protected void initializeDesktopProperties();

    public void addPropertyChangeListener(String name, PropertyChangeListener pcl);

    public void removePropertyChangeListener(String name, PropertyChangeListener pcl);

    public PropertyChangeListener[] getPropertyChangeListeners();

    public PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    protected final Map<String, Object> desktopProperties;

    protected final PropertyChangeSupport desktopPropsSupport;

    public boolean isAlwaysOnTopSupported();

    public abstract boolean isModalityTypeSupported(Dialog.ModalityType modalityType);

    public abstract boolean isModalExclusionTypeSupported(Dialog.ModalExclusionType modalExclusionType);

    public void addAWTEventListener(AWTEventListener listener, long eventMask);

    public void removeAWTEventListener(AWTEventListener listener);

    static boolean enabledOnToolkit(long eventMask);

    synchronized int countAWTEventListeners(long eventMask);

    public AWTEventListener[] getAWTEventListeners();

    public AWTEventListener[] getAWTEventListeners(long eventMask);

    void notifyAWTEventListeners(AWTEvent theEvent);

    private static class ToolkitEventMulticaster extends AWTEventMulticaster implements AWTEventListener {

        @SuppressWarnings("overloads")
        static AWTEventListener add(AWTEventListener a, AWTEventListener b);

        @SuppressWarnings("overloads")
        static AWTEventListener remove(AWTEventListener l, AWTEventListener oldl);

        protected EventListener remove(EventListener oldl);

        public void eventDispatched(AWTEvent event);
    }

    private class SelectiveAWTEventListener implements AWTEventListener {

        public AWTEventListener getListener();

        public long getEventMask();

        public int[] getCalls();

        public void orEventMasks(long mask);

        public void eventDispatched(AWTEvent event);
    }

    public abstract Map<java.awt.font.TextAttribute, ?> mapInputMethodHighlight(InputMethodHighlight highlight) throws HeadlessException;

    @SuppressWarnings("serial")
    private static class DesktopPropertyChangeSupport extends PropertyChangeSupport {

        public DesktopPropertyChangeSupport(Object sourceBean) {
        }

        @Override
        public synchronized void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

        @Override
        public synchronized void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

        @Override
        public synchronized PropertyChangeListener[] getPropertyChangeListeners();

        @Override
        public synchronized PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

        @Override
        public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

        @Override
        public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

        @Override
        public void firePropertyChange(final PropertyChangeEvent evt);
    }

    public boolean areExtraMouseButtonsEnabled() throws HeadlessException;
}

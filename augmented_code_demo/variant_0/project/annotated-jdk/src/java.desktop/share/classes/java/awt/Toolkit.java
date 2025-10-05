/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.datatransfer.Clipboard;
    @Positive
import java.awt.dnd.DragGestureListener;
    @Positive
import java.awt.dnd.DragGestureRecognizer;
    @Positive
import java.awt.dnd.DragSource;
    @Positive
import java.awt.event.AWTEventListener;
    @Positive
import java.awt.event.AWTEventListenerProxy;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.AdjustmentEvent;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.ContainerEvent;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.HierarchyEvent;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.InputMethodEvent;
    @Positive
import java.awt.event.InvocationEvent;
    @Positive
import java.awt.event.ItemEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.awt.event.PaintEvent;
    @Positive
import java.awt.event.TextEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.im.InputMethodHighlight;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.image.ImageObserver;
    @Positive
import java.awt.image.ImageProducer;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.PropertyChangeSupport;
    @Positive
import java.io.File;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.util.Properties;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.accessibility.AccessibilityProvider;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTPermissions;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.HeadlessToolkit;
    @Positive
import sun.awt.PeerEvent;
    @Positive
import sun.awt.PlatformGraphicsInfo;
    @Positive
import sun.awt.SunToolkit;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Toolkit {

    @Positive
    protected Toolkit() {
    @Positive
    }

    @Positive
    protected void loadSystemColors(int[] systemColors) throws HeadlessException;

    @Positive
    public void setDynamicLayout(final boolean dynamic) throws HeadlessException;

    @Positive
    protected boolean isDynamicLayoutSet() throws HeadlessException;

    @Positive
    public boolean isDynamicLayoutActive() throws HeadlessException;

    @Positive
    public abstract Dimension getScreenSize() throws HeadlessException;

    @Positive
    public abstract int getScreenResolution() throws HeadlessException;

    @Positive
    public Insets getScreenInsets(GraphicsConfiguration gc) throws HeadlessException;

    @Positive
    public abstract ColorModel getColorModel() throws HeadlessException;

    @Positive
    @Deprecated
    @Positive
    public abstract String[] getFontList();

    @Positive
    @Deprecated
    @Positive
    public abstract FontMetrics getFontMetrics(Font font);

    @Positive
    public abstract void sync();

    @Positive
    public static synchronized Toolkit getDefaultToolkit();

    @Positive
    public abstract Image getImage(String filename);

    @Positive
    public abstract Image getImage(URL url);

    @Positive
    public abstract Image createImage(String filename);

    @Positive
    public abstract Image createImage(URL url);

    @Positive
    public abstract boolean prepareImage(Image image, int width, int height, ImageObserver observer);

    @Positive
    public abstract int checkImage(Image image, int width, int height, ImageObserver observer);

    @Positive
    public abstract Image createImage(ImageProducer producer);

    @Positive
    public Image createImage(byte[] imagedata);

    @Positive
    public abstract Image createImage(byte[] imagedata, int imageoffset, int imagelength);

    @Positive
    public abstract PrintJob getPrintJob(Frame frame, String jobtitle, Properties props);

    @Positive
    public PrintJob getPrintJob(Frame frame, String jobtitle, JobAttributes jobAttributes, PageAttributes pageAttributes);

    @Positive
    public abstract void beep();

    @Positive
    public abstract Clipboard getSystemClipboard() throws HeadlessException;

    @Positive
    public Clipboard getSystemSelection() throws HeadlessException;

    @Positive
    @Deprecated()
    @Positive
    public int getMenuShortcutKeyMask() throws HeadlessException;

    @Positive
    public int getMenuShortcutKeyMaskEx() throws HeadlessException;

    @Positive
    public boolean getLockingKeyState(int keyCode) throws UnsupportedOperationException;

    @Positive
    public void setLockingKeyState(int keyCode, boolean on) throws UnsupportedOperationException;

    @Positive
    protected static Container getNativeContainer(Component c);

    @Positive
    public Cursor createCustomCursor(Image cursor, Point hotSpot, String name) throws IndexOutOfBoundsException, HeadlessException;

    @Positive
    public Dimension getBestCursorSize(int preferredWidth, int preferredHeight) throws HeadlessException;

    @Positive
    public int getMaximumCursorColors() throws HeadlessException;

    @Positive
    public boolean isFrameStateSupported(int state) throws HeadlessException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static void loadLibraries();

    @Positive
    public static String getProperty(String key, String defaultValue);

    @Positive
    public final EventQueue getSystemEventQueue();

    @Positive
    protected abstract EventQueue getSystemEventQueueImpl();

    @Positive
    static EventQueue getEventQueue();

    @Positive
    public <T extends DragGestureRecognizer> T createDragGestureRecognizer(Class<T> abstractRecognizerClass, DragSource ds, Component c, int srcActions, DragGestureListener dgl);

    @Positive
    public final synchronized Object getDesktopProperty(String propertyName);

    @Positive
    protected final void setDesktopProperty(String name, Object newValue);

    @Positive
    protected Object lazilyLoadDesktopProperty(String name);

    @Positive
    protected void initializeDesktopProperties();

    @Positive
    public void addPropertyChangeListener(String name, PropertyChangeListener pcl);

    @Positive
    public void removePropertyChangeListener(String name, PropertyChangeListener pcl);

    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
    public PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    @Positive
    protected final Map<String, Object> desktopProperties;

    @Positive
    protected final PropertyChangeSupport desktopPropsSupport;

    @Positive
    public boolean isAlwaysOnTopSupported();

    @Positive
    public abstract boolean isModalityTypeSupported(Dialog.ModalityType modalityType);

    @Positive
    public abstract boolean isModalExclusionTypeSupported(Dialog.ModalExclusionType modalExclusionType);

    @Positive
    public void addAWTEventListener(AWTEventListener listener, long eventMask);

    @Positive
    public void removeAWTEventListener(AWTEventListener listener);

    @Positive
    static boolean enabledOnToolkit(long eventMask);

    @Positive
    synchronized int countAWTEventListeners(long eventMask);

    @Positive
    public AWTEventListener[] getAWTEventListeners();

    @Positive
    public AWTEventListener[] getAWTEventListeners(long eventMask);

    @Positive
    void notifyAWTEventListeners(AWTEvent theEvent);

    @Positive
    private static class ToolkitEventMulticaster extends AWTEventMulticaster implements AWTEventListener {

    @Positive
        @SuppressWarnings("overloads")
    @Positive
        static AWTEventListener add(AWTEventListener a, AWTEventListener b);

    @Positive
        @SuppressWarnings("overloads")
    @Positive
        static AWTEventListener remove(AWTEventListener l, AWTEventListener oldl);

    @Positive
        protected EventListener remove(EventListener oldl);

    @Positive
        public void eventDispatched(AWTEvent event);
    @Positive
    }

    @Positive
    private class SelectiveAWTEventListener implements AWTEventListener {

    @Positive
        public AWTEventListener getListener();

    @Positive
        public long getEventMask();

    @Positive
        public int[] getCalls();

    @Positive
        public void orEventMasks(long mask);

    @Positive
        public void eventDispatched(AWTEvent event);
    @Positive
    }

    @Positive
    public abstract Map<java.awt.font.TextAttribute, ?> mapInputMethodHighlight(InputMethodHighlight highlight) throws HeadlessException;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    private static class DesktopPropertyChangeSupport extends PropertyChangeSupport {

    @Positive
        public DesktopPropertyChangeSupport(Object sourceBean) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public synchronized void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
        @Override
    @Positive
        public synchronized void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
        @Override
    @Positive
        public synchronized PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
        @Override
    @Positive
        public synchronized PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    @Positive
        @Override
    @Positive
        public synchronized void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
        @Override
    @Positive
        public synchronized void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
        @Override
    @Positive
        public void firePropertyChange(final PropertyChangeEvent evt);
    @Positive
    }

    @Positive
    public boolean areExtraMouseButtonsEnabled() throws HeadlessException;
    @Positive
}

// CFWR semantic augmentation - variant 0

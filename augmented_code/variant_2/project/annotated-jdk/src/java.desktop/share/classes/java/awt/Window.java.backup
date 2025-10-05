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

import java.awt.event.ComponentEvent;
import java.awt.event.FocusEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.WindowEvent;
import java.awt.event.WindowFocusListener;
import java.awt.event.WindowListener;
import java.awt.event.WindowStateListener;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.im.InputContext;
import java.awt.image.BufferStrategy;
import java.awt.peer.ComponentPeer;
import java.awt.peer.WindowPeer;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OptionalDataException;
import java.io.Serial;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.lang.reflect.InvocationTargetException;
import java.security.AccessController;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EventListener;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import sun.awt.AWTAccessor;
import sun.awt.AWTPermissions;
import sun.awt.AppContext;
import sun.awt.DebugSettings;
import sun.awt.SunToolkit;
import sun.awt.util.IdentityArrayList;
import sun.java2d.pipe.Region;
import sun.security.action.GetPropertyAction;
import sun.util.logging.PlatformLogger;

@AnnotatedFor({ "nullness" })
public class Window extends Container implements Accessible {

    public static enum Type {

        NORMAL, UTILITY, POPUP
    }

    static class WindowDisposerRecord implements sun.java2d.DisposerRecord {

        public void updateOwner();

        public void dispose();
    }

    public Window(@Nullable Frame owner) {
    }

    public Window(@Nullable Window owner) {
    }

    public Window(@Nullable Window owner, @Nullable GraphicsConfiguration gc) {
    }

    String constructComponentName();

    public java.util.List<Image> getIconImages();

    public synchronized void setIconImages(java.util.@Nullable List<? extends Image> icons);

    public void setIconImage(@Nullable Image image);

    public void addNotify();

    public void removeNotify();

    @SuppressWarnings("deprecation")
    public void pack();

    public void setMinimumSize(@Nullable Dimension minimumSize);

    public void setSize(Dimension d);

    public void setSize(int width, int height);

    @Override
    public void setLocation(int x, int y);

    @Override
    public void setLocation(Point p);

    @Deprecated
    public void reshape(int x, int y, int width, int height);

    void setClientSize(int w, int h);

    final void closeSplashScreen();

    public void setVisible(boolean b);

    @Deprecated
    public void show();

    static void updateChildFocusableWindowState(Window w);

    synchronized void postWindowEvent(int id);

    @Deprecated
    public void hide();

    final void clearMostRecentFocusOwnerOnHide();

    public void dispose();

    void disposeImpl();

    void doDispose();

    void adjustListeningChildrenOnParent(long mask, int num);

    void adjustDescendantsOnParent(int num);

    public void toFront();

    final void toFront_NoClientCode();

    public void toBack();

    final void toBack_NoClientCode();

    public Toolkit getToolkit();

    @Nullable
    public final String getWarningString();

    public Locale getLocale();

    public InputContext getInputContext();

    public void setCursor(@Nullable Cursor cursor);

    @Nullable
    public Window getOwner();

    @Nullable
    final Window getOwner_NoClientCode();

    public Window[] getOwnedWindows();

    final Window[] getOwnedWindows_NoClientCode();

    boolean isModalBlocked();

    void setModalBlocked(Dialog blocker, boolean blocked, boolean peerCall);

    @Nullable
    Dialog getModalBlocker();

    static IdentityArrayList<Window> getAllWindows();

    static IdentityArrayList<Window> getAllUnblockedWindows();

    public static Window[] getWindows();

    public static Window[] getOwnerlessWindows();

    Window getDocumentRoot();

    public void setModalExclusionType(Dialog.ModalExclusionType exclusionType);

    public Dialog.ModalExclusionType getModalExclusionType();

    boolean isModalExcluded(Dialog.ModalExclusionType exclusionType);

    void updateChildrenBlocking();

    public synchronized void addWindowListener(WindowListener l);

    public synchronized void addWindowStateListener(WindowStateListener l);

    public synchronized void addWindowFocusListener(WindowFocusListener l);

    public synchronized void removeWindowListener(WindowListener l);

    public synchronized void removeWindowStateListener(WindowStateListener l);

    public synchronized void removeWindowFocusListener(WindowFocusListener l);

    public synchronized WindowListener[] getWindowListeners();

    public synchronized WindowFocusListener[] getWindowFocusListeners();

    public synchronized WindowStateListener[] getWindowStateListeners();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    boolean eventEnabled(AWTEvent e);

    protected void processEvent(AWTEvent e);

    protected void processWindowEvent(WindowEvent e);

    protected void processWindowFocusEvent(WindowEvent e);

    protected void processWindowStateEvent(WindowEvent e);

    void preProcessKeyEvent(KeyEvent e);

    void postProcessKeyEvent(KeyEvent e);

    public final void setAlwaysOnTop(boolean alwaysOnTop) throws SecurityException;

    public boolean isAlwaysOnTopSupported();

    public final boolean isAlwaysOnTop();

    @Nullable
    public Component getFocusOwner();

    @Nullable
    public Component getMostRecentFocusOwner();

    public boolean isActive();

    public boolean isFocused();

    @SuppressWarnings("unchecked")
    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    public final void setFocusCycleRoot(boolean focusCycleRoot);

    public final boolean isFocusCycleRoot();

    @Nullable
    public final Container getFocusCycleRootAncestor();

    public final boolean isFocusableWindow();

    public boolean getFocusableWindowState();

    public void setFocusableWindowState(boolean focusableWindowState);

    public void setAutoRequestFocus(boolean autoRequestFocus);

    public boolean isAutoRequestFocus();

    public void addPropertyChangeListener(PropertyChangeListener listener);

    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Override
    public boolean isValidateRoot();

    void dispatchEventImpl(AWTEvent e);

    @Deprecated
    public boolean postEvent(Event e);

    public boolean isShowing();

    boolean isDisposing();

    @Deprecated
    public void applyResourceBundle(ResourceBundle rb);

    @Deprecated
    public void applyResourceBundle(String rbName);

    void addOwnedWindow(WeakReference<Window> weakWindow);

    void removeOwnedWindow(WeakReference<Window> weakWindow);

    void connectOwnedWindow(Window child);

    public void setType(Type type);

    public Type getType();

    public AccessibleContext getAccessibleContext();

    protected class AccessibleAWTWindow extends AccessibleAWTContainer {

        protected AccessibleAWTWindow() {
        }

        public AccessibleRole getAccessibleRole();

        public AccessibleStateSet getAccessibleStateSet();
    }

    @Override
    void setGraphicsConfiguration(@Nullable GraphicsConfiguration gc);

    public void setLocationRelativeTo(@Nullable Component c);

    void deliverMouseWheelToAncestor(MouseWheelEvent e);

    boolean dispatchMouseWheelToAncestor(MouseWheelEvent e);

    public void createBufferStrategy(int numBuffers);

    public void createBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException;

    @Nullable
    public BufferStrategy getBufferStrategy();

    Component getTemporaryLostComponent();

    Component setTemporaryLostComponent(Component component);

    boolean canContainFocusOwner(Component focusOwnerCandidate);

    public void setLocationByPlatform(boolean locationByPlatform);

    public boolean isLocationByPlatform();

    public void setBounds(int x, int y, int width, int height);

    public void setBounds(Rectangle r);

    boolean isRecursivelyVisible();

    public float getOpacity();

    @SuppressWarnings("deprecation")
    public void setOpacity(float opacity);

    @Nullable
    public Shape getShape();

    public void setShape(@Nullable Shape shape);

    @Override
    @Nullable
    public Color getBackground();

    @Override
    public void setBackground(@Nullable Color bgColor);

    @Override
    public boolean isOpaque();

    @Override
    public void paint(Graphics g);

    @Override
    @Nullable
    final Container getContainer();

    @Override
    final void applyCompoundShape(Region shape);

    @Override
    final void applyCurrentShape();

    @Override
    final void mixOnReshaping();

    @Override
    final Point getLocationOnWindow();

    @Override
    void updateZOrder();
}

class FocusManager implements java.io.Serializable {
}

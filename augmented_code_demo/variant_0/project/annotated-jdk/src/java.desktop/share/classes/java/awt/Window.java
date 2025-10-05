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
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.MouseWheelEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.event.WindowFocusListener;
    @Positive
import java.awt.event.WindowListener;
    @Positive
import java.awt.event.WindowStateListener;
    @Positive
import java.awt.geom.Path2D;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.im.InputContext;
    @Positive
import java.awt.image.BufferStrategy;
    @Positive
import java.awt.peer.ComponentPeer;
    @Positive
import java.awt.peer.WindowPeer;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.OptionalDataException;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.security.AccessController;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.Locale;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Set;
    @Positive
import java.util.Vector;
    @Positive
import java.util.concurrent.atomic.AtomicBoolean;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTPermissions;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.DebugSettings;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.util.IdentityArrayList;
    @Positive
import sun.java2d.pipe.Region;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class Window extends Container implements Accessible {

    @Positive
    public static enum Type {

    @Positive
        NORMAL, UTILITY, POPUP
    @Positive
    }

    @Positive
    static class WindowDisposerRecord implements sun.java2d.DisposerRecord {

    @Positive
        public void updateOwner();

    @Positive
        public void dispose();
    @Positive
    }

    @Positive
    public Window(@Nullable Frame owner) {
    @Positive
    }

    @Positive
    public Window(@Nullable Window owner) {
    @Positive
    }

    @Positive
    public Window(@Nullable Window owner, @Nullable GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    String constructComponentName();

    @Positive
    public java.util.List<Image> getIconImages();

    @Positive
    public synchronized void setIconImages(java.util.@Nullable List<? extends Image> icons);

    @Positive
    public void setIconImage(@Nullable Image image);

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void pack();

    @Positive
    public void setMinimumSize(@Nullable Dimension minimumSize);

    @Positive
    public void setSize(Dimension d);

    @Positive
    public void setSize(int width, int height);

    @Positive
    @Override
    @Positive
    public void setLocation(int x, int y);

    @Positive
    @Override
    @Positive
    public void setLocation(Point p);

    @Positive
    @Deprecated
    @Positive
    public void reshape(int x, int y, int width, int height);

    @Positive
    void setClientSize(int w, int h);

    @Positive
    final void closeSplashScreen();

    @Positive
    public void setVisible(boolean b);

    @Positive
    @Deprecated
    @Positive
    public void show();

    @Positive
    static void updateChildFocusableWindowState(Window w);

    @Positive
    synchronized void postWindowEvent(int id);

    @Positive
    @Deprecated
    @Positive
    public void hide();

    @Positive
    final void clearMostRecentFocusOwnerOnHide();

    @Positive
    public void dispose();

    @Positive
    void disposeImpl();

    @Positive
    void doDispose();

    @Positive
    void adjustListeningChildrenOnParent(long mask, int num);

    @Positive
    void adjustDescendantsOnParent(int num);

    @Positive
    public void toFront();

    @Positive
    final void toFront_NoClientCode();

    @Positive
    public void toBack();

    @Positive
    final void toBack_NoClientCode();

    @Positive
    public Toolkit getToolkit();

    @Positive
    @Nullable
    @Positive
    public final String getWarningString();

    @Positive
    public Locale getLocale();

    @Positive
    public InputContext getInputContext();

    @Positive
    public void setCursor(@Nullable Cursor cursor);

    @Positive
    @Nullable
    @Positive
    public Window getOwner();

    @Positive
    @Nullable
    @Positive
    final Window getOwner_NoClientCode();

    @Positive
    public Window[] getOwnedWindows();

    @Positive
    final Window[] getOwnedWindows_NoClientCode();

    @Positive
    boolean isModalBlocked();

    @Positive
    void setModalBlocked(Dialog blocker, boolean blocked, boolean peerCall);

    @Positive
    @Nullable
    @Positive
    Dialog getModalBlocker();

    @Positive
    static IdentityArrayList<Window> getAllWindows();

    @Positive
    static IdentityArrayList<Window> getAllUnblockedWindows();

    @Positive
    public static Window[] getWindows();

    @Positive
    public static Window[] getOwnerlessWindows();

    @Positive
    Window getDocumentRoot();

    @Positive
    public void setModalExclusionType(Dialog.ModalExclusionType exclusionType);

    @Positive
    public Dialog.ModalExclusionType getModalExclusionType();

    @Positive
    boolean isModalExcluded(Dialog.ModalExclusionType exclusionType);

    @Positive
    void updateChildrenBlocking();

    @Positive
    public synchronized void addWindowListener(WindowListener l);

    @Positive
    public synchronized void addWindowStateListener(WindowStateListener l);

    @Positive
    public synchronized void addWindowFocusListener(WindowFocusListener l);

    @Positive
    public synchronized void removeWindowListener(WindowListener l);

    @Positive
    public synchronized void removeWindowStateListener(WindowStateListener l);

    @Positive
    public synchronized void removeWindowFocusListener(WindowFocusListener l);

    @Positive
    public synchronized WindowListener[] getWindowListeners();

    @Positive
    public synchronized WindowFocusListener[] getWindowFocusListeners();

    @Positive
    public synchronized WindowStateListener[] getWindowStateListeners();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected void processWindowEvent(WindowEvent e);

    @Positive
    protected void processWindowFocusEvent(WindowEvent e);

    @Positive
    protected void processWindowStateEvent(WindowEvent e);

    @Positive
    void preProcessKeyEvent(KeyEvent e);

    @Positive
    void postProcessKeyEvent(KeyEvent e);

    @Positive
    public final void setAlwaysOnTop(boolean alwaysOnTop) throws SecurityException;

    @Positive
    public boolean isAlwaysOnTopSupported();

    @Positive
    public final boolean isAlwaysOnTop();

    @Positive
    @Nullable
    @Positive
    public Component getFocusOwner();

    @Positive
    @Nullable
    @Positive
    public Component getMostRecentFocusOwner();

    @Positive
    public boolean isActive();

    @Positive
    public boolean isFocused();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    @Positive
    public final void setFocusCycleRoot(boolean focusCycleRoot);

    @Positive
    public final boolean isFocusCycleRoot();

    @Positive
    @Nullable
    @Positive
    public final Container getFocusCycleRootAncestor();

    @Positive
    public final boolean isFocusableWindow();

    @Positive
    public boolean getFocusableWindowState();

    @Positive
    public void setFocusableWindowState(boolean focusableWindowState);

    @Positive
    public void setAutoRequestFocus(boolean autoRequestFocus);

    @Positive
    public boolean isAutoRequestFocus();

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    @Override
    @Positive
    public boolean isValidateRoot();

    @Positive
    void dispatchEventImpl(AWTEvent e);

    @Positive
    @Deprecated
    @Positive
    public boolean postEvent(Event e);

    @Positive
    public boolean isShowing();

    @Positive
    boolean isDisposing();

    @Positive
    @Deprecated
    @Positive
    public void applyResourceBundle(ResourceBundle rb);

    @Positive
    @Deprecated
    @Positive
    public void applyResourceBundle(String rbName);

    @Positive
    void addOwnedWindow(WeakReference<Window> weakWindow);

    @Positive
    void removeOwnedWindow(WeakReference<Window> weakWindow);

    @Positive
    void connectOwnedWindow(Window child);

    @Positive
    public void setType(Type type);

    @Positive
    public Type getType();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleAWTWindow extends AccessibleAWTContainer {

    @Positive
        protected AccessibleAWTWindow() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }

    @Positive
    @Override
    @Positive
    void setGraphicsConfiguration(@Nullable GraphicsConfiguration gc);

    @Positive
    public void setLocationRelativeTo(@Nullable Component c);

    @Positive
    void deliverMouseWheelToAncestor(MouseWheelEvent e);

    @Positive
    boolean dispatchMouseWheelToAncestor(MouseWheelEvent e);

    @Positive
    public void createBufferStrategy(int numBuffers);

    @Positive
    public void createBufferStrategy(int numBuffers, BufferCapabilities caps) throws AWTException;

    @Positive
    @Nullable
    @Positive
    public BufferStrategy getBufferStrategy();

    @Positive
    Component getTemporaryLostComponent();

    @Positive
    Component setTemporaryLostComponent(Component component);

    @Positive
    boolean canContainFocusOwner(Component focusOwnerCandidate);

    @Positive
    public void setLocationByPlatform(boolean locationByPlatform);

    @Positive
    public boolean isLocationByPlatform();

    @Positive
    public void setBounds(int x, int y, int width, int height);

    @Positive
    public void setBounds(Rectangle r);

    @Positive
    boolean isRecursivelyVisible();

    @Positive
    public float getOpacity();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void setOpacity(float opacity);

    @Positive
    @Nullable
    @Positive
    public Shape getShape();

    @Positive
    public void setShape(@Nullable Shape shape);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public Color getBackground();

    @Positive
    @Override
    @Positive
    public void setBackground(@Nullable Color bgColor);

    @Positive
    @Override
    @Positive
    public boolean isOpaque();

    @Positive
    @Override
    @Positive
    public void paint(Graphics g);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    final Container getContainer();

    @Positive
    @Override
    @Positive
    final void applyCompoundShape(Region shape);

    @Positive
    @Override
    @Positive
    final void applyCurrentShape();

    @Positive
    @Override
    @Positive
    final void mixOnReshaping();

    @Positive
    @Override
    @Positive
    final Point getLocationOnWindow();

    @Positive
    @Override
    @Positive
    void updateZOrder();
    @Positive
}

    @Positive
class FocusManager implements java.io.Serializable {
    @Positive
}

// CFWR semantic augmentation - variant 0

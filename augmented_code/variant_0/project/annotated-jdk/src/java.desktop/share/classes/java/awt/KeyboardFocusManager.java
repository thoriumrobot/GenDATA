/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.peer.KeyboardFocusManagerPeer;
    @Positive
import java.awt.peer.LightweightPeer;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.PropertyChangeSupport;
    @Positive
import java.beans.PropertyVetoException;
    @Positive
import java.beans.VetoableChangeListener;
    @Positive
import java.beans.VetoableChangeSupport;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedList;
    @Positive
import java.util.Set;
    @Positive
import java.util.StringTokenizer;
    @Positive
import java.util.WeakHashMap;
    @Positive
import sun.util.logging.PlatformLogger;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.KeyboardFocusManagerPeerProvider;
    @Positive
import sun.awt.AWTAccessor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class KeyboardFocusManager implements KeyEventDispatcher, KeyEventPostProcessor {

    @Positive
    public static final int FORWARD_TRAVERSAL_KEYS;

    @Positive
    public static final int BACKWARD_TRAVERSAL_KEYS;

    @Positive
    public static final int UP_CYCLE_TRAVERSAL_KEYS;

    @Positive
    public static final int DOWN_CYCLE_TRAVERSAL_KEYS;

    @Positive
    public static KeyboardFocusManager getCurrentKeyboardFocusManager();

    @Positive
    static synchronized KeyboardFocusManager getCurrentKeyboardFocusManager(AppContext appcontext);

    @Positive
    public static void setCurrentKeyboardFocusManager(KeyboardFocusManager newManager) throws SecurityException;

    @Positive
    final void setCurrentSequencedEvent(SequencedEvent current);

    @Positive
    final SequencedEvent getCurrentSequencedEvent();

    @Positive
    static Set<AWTKeyStroke> initFocusTraversalKeysSet(String value, Set<AWTKeyStroke> targetSet);

    @Positive
    public KeyboardFocusManager() {
    @Positive
    }

    @Positive
    public Component getFocusOwner();

    @Positive
    protected Component getGlobalFocusOwner() throws SecurityException;

    @Positive
    protected void setGlobalFocusOwner(Component focusOwner) throws SecurityException;

    @Positive
    public void clearFocusOwner();

    @Positive
    public void clearGlobalFocusOwner() throws SecurityException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    void clearGlobalFocusOwnerPriv();

    @Positive
    Component getNativeFocusOwner();

    @Positive
    void setNativeFocusOwner(Component comp);

    @Positive
    Window getNativeFocusedWindow();

    @Positive
    public Component getPermanentFocusOwner();

    @Positive
    protected Component getGlobalPermanentFocusOwner() throws SecurityException;

    @Positive
    protected void setGlobalPermanentFocusOwner(Component permanentFocusOwner) throws SecurityException;

    @Positive
    public Window getFocusedWindow();

    @Positive
    protected Window getGlobalFocusedWindow() throws SecurityException;

    @Positive
    protected void setGlobalFocusedWindow(Window focusedWindow) throws SecurityException;

    @Positive
    public Window getActiveWindow();

    @Positive
    protected Window getGlobalActiveWindow() throws SecurityException;

    @Positive
    protected void setGlobalActiveWindow(Window activeWindow) throws SecurityException;

    @Positive
    public synchronized FocusTraversalPolicy getDefaultFocusTraversalPolicy();

    @Positive
    public void setDefaultFocusTraversalPolicy(FocusTraversalPolicy defaultPolicy);

    @Positive
    public void setDefaultFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    @Positive
    public Set<AWTKeyStroke> getDefaultFocusTraversalKeys(int id);

    @Positive
    public Container getCurrentFocusCycleRoot();

    @Positive
    protected Container getGlobalCurrentFocusCycleRoot() throws SecurityException;

    @Positive
    public void setGlobalCurrentFocusCycleRoot(Container newFocusCycleRoot) throws SecurityException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    void setGlobalCurrentFocusCycleRootPriv(final Container newFocusCycleRoot);

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public synchronized PropertyChangeListener[] getPropertyChangeListeners();

    @Positive
    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    public void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    public synchronized PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    @Positive
    protected void firePropertyChange(String propertyName, Object oldValue, Object newValue);

    @Positive
    public void addVetoableChangeListener(VetoableChangeListener listener);

    @Positive
    public void removeVetoableChangeListener(VetoableChangeListener listener);

    @Positive
    public synchronized VetoableChangeListener[] getVetoableChangeListeners();

    @Positive
    public void addVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    @Positive
    public void removeVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    @Positive
    public synchronized VetoableChangeListener[] getVetoableChangeListeners(String propertyName);

    @Positive
    protected void fireVetoableChange(String propertyName, Object oldValue, Object newValue) throws PropertyVetoException;

    @Positive
    public void addKeyEventDispatcher(KeyEventDispatcher dispatcher);

    @Positive
    public void removeKeyEventDispatcher(KeyEventDispatcher dispatcher);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    protected synchronized java.util.List<KeyEventDispatcher> getKeyEventDispatchers();

    @Positive
    public void addKeyEventPostProcessor(KeyEventPostProcessor processor);

    @Positive
    public void removeKeyEventPostProcessor(KeyEventPostProcessor processor);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    protected java.util.List<KeyEventPostProcessor> getKeyEventPostProcessors();

    @Positive
    static void setMostRecentFocusOwner(Component component);

    @Positive
    static synchronized void setMostRecentFocusOwner(Window window, Component component);

    @Positive
    static void clearMostRecentFocusOwner(Component comp);

    @Positive
    static synchronized Component getMostRecentFocusOwner(Window window);

    @Positive
    public abstract boolean dispatchEvent(AWTEvent e);

    @Positive
    public final void redispatchEvent(Component target, AWTEvent e);

    @Positive
    public abstract boolean dispatchKeyEvent(KeyEvent e);

    @Positive
    public abstract boolean postProcessKeyEvent(KeyEvent e);

    @Positive
    public abstract void processKeyEvent(Component focusedComponent, KeyEvent e);

    @Positive
    protected abstract void enqueueKeyEvents(long after, Component untilFocused);

    @Positive
    protected abstract void dequeueKeyEvents(long after, Component untilFocused);

    @Positive
    protected abstract void discardKeyEvents(Component comp);

    @Positive
    public abstract void focusNextComponent(Component aComponent);

    @Positive
    public abstract void focusPreviousComponent(Component aComponent);

    @Positive
    public abstract void upFocusCycle(Component aComponent);

    @Positive
    public abstract void downFocusCycle(Container aContainer);

    @Positive
    public final void focusNextComponent();

    @Positive
    public final void focusPreviousComponent();

    @Positive
    public final void upFocusCycle();

    @Positive
    public final void downFocusCycle();

    @Positive
    void dumpRequests();

    @Positive
    private static final class LightweightFocusRequest {

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static final class HeavyweightFocusRequest {

    @Positive
        boolean addLightweightRequest(Component descendant, boolean temporary, FocusEvent.Cause cause);

    @Positive
        LightweightFocusRequest getFirstLightweightRequest();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static boolean processSynchronousLightweightTransfer(Component heavyweight, Component descendant, boolean temporary, boolean focusedWindowChangeAllowed, long time);

    @Positive
    static int shouldNativelyFocusHeavyweight(Component heavyweight, Component descendant, boolean temporary, boolean focusedWindowChangeAllowed, long time, FocusEvent.Cause cause);

    @Positive
    static Window markClearGlobalFocusOwner();

    @Positive
    Component getCurrentWaitingRequest(Component parent);

    @Positive
    static boolean isAutoFocusTransferEnabled();

    @Positive
    static boolean isAutoFocusTransferEnabledFor(Component comp);

    @Positive
    static void processCurrentLightweightRequests();

    @Positive
    static FocusEvent retargetUnexpectedFocusEvent(FocusEvent fe);

    @Positive
    static FocusEvent retargetFocusGained(FocusEvent fe);

    @Positive
    static FocusEvent retargetFocusLost(FocusEvent fe);

    @Positive
    static AWTEvent retargetFocusEvent(AWTEvent event);

    @Positive
    void clearMarkers();

    @Positive
    static boolean removeFirstRequest();

    @Positive
    static void removeLastFocusRequest(Component heavyweight);

    @Positive
    static Component getHeavyweight(Component comp);

    @Positive
    static boolean isProxyActive(KeyEvent e);
    @Positive
}

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
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.dnd.DropTarget;
    @Positive
import java.awt.event.AWTEventListener;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.ContainerEvent;
    @Positive
import java.awt.event.ContainerListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.HierarchyEvent;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.awt.event.MouseWheelEvent;
    @Positive
import java.awt.peer.ComponentPeer;
    @Positive
import java.awt.peer.ContainerPeer;
    @Positive
import java.awt.peer.LightweightPeer;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Set;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleComponent;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTAccessor.MouseEventAccessor;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.PeerEvent;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.dnd.SunDropTargetEvent;
    @Positive
import sun.java2d.pipe.Region;
    @Positive
import sun.security.action.GetBooleanAction;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
@AnnotatedFor({ "guieffect", "nullness" })
    @Positive
@UIType
    @Positive
public class Container extends Component {

    @Positive
    public Container() {
    @Positive
    }

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    void initializeFocusTraversalKeys();

    @Positive
    public int getComponentCount();

    @Positive
    @Deprecated
    @Positive
    public int countComponents();

    @Positive
    public Component getComponent(int n);

    @Positive
    public Component[] getComponents();

    @Positive
    final Component[] getComponents_NoClientCode();

    @Positive
    Component[] getComponentsSync();

    @Positive
    public Insets getInsets();

    @Positive
    @Deprecated
    @Positive
    public Insets insets();

    @Positive
    public Component add(@UnknownInitialization(Container.class) Container this, Component comp);

    @Positive
    public Component add(@UnknownInitialization(Container.class) Container this, @Nullable String name, Component comp);

    @Positive
    public Component add(@UnknownInitialization(Container.class) Container this, Component comp, int index);

    @Positive
    boolean canContainFocusOwner(Component focusOwnerCandidate);

    @Positive
    final boolean hasHeavyweightDescendants();

    @Positive
    final boolean hasLightweightDescendants();

    @Positive
    Container getHeavyweightContainer();

    @Positive
    public void setComponentZOrder(Component comp, int index);

    @Positive
    public int getComponentZOrder(Component comp);

    @Positive
    public void add(@UnknownInitialization(Container.class) Container this, Component comp, @Nullable Object constraints);

    @Positive
    public void add(@UnknownInitialization(Container.class) Container this, Component comp, @Nullable Object constraints, int index);

    @Positive
    protected void addImpl(Component comp, @Nullable Object constraints, int index);

    @Positive
    @Override
    @Positive
    final boolean updateChildGraphicsData(GraphicsConfiguration gc);

    @Positive
    void checkGD(String stringID);

    @Positive
    public void remove(int index);

    @Positive
    public void remove(Component comp);

    @Positive
    public void removeAll();

    @Positive
    int numListening(long mask);

    @Positive
    void adjustListeningChildren(long mask, int num);

    @Positive
    void adjustDescendants(int num);

    @Positive
    void adjustDescendantsOnParent(int num);

    @Positive
    int countHierarchyMembers();

    @Positive
    final int createHierarchyEvents(int id, Component changed, Container changedParent, long changeFlags, boolean enabledOnToolkit);

    @Positive
    final void createChildHierarchyEvents(int id, long changeFlags, boolean enabledOnToolkit);

    @Positive
    @Nullable
    @Positive
    public LayoutManager getLayout();

    @Positive
    public void setLayout(@UnknownInitialization(Container.class) Container this, @Nullable LayoutManager mgr);

    @Positive
    public void doLayout();

    @Positive
    @Deprecated
    @Positive
    public void layout();

    @Positive
    public boolean isValidateRoot();

    @Positive
    @Override
    @Positive
    void invalidateParent();

    @Positive
    @SafeEffect
    @Positive
    @Override
    @Positive
    public void invalidate();

    @Positive
    public void validate();

    @Positive
    final void validateUnconditionally();

    @Positive
    protected void validateTree();

    @Positive
    void invalidateTree();

    @Positive
    public void setFont(@Nullable Font f);

    @Positive
    public Dimension getPreferredSize();

    @Positive
    @Deprecated
    @Positive
    public Dimension preferredSize();

    @Positive
    public Dimension getMinimumSize();

    @Positive
    @Deprecated
    @Positive
    public Dimension minimumSize();

    @Positive
    public Dimension getMaximumSize();

    @Positive
    public float getAlignmentX();

    @Positive
    public float getAlignmentY();

    @Positive
    public void paint(Graphics g);

    @Positive
    public void update(Graphics g);

    @Positive
    public void print(Graphics g);

    @Positive
    public void paintComponents(Graphics g);

    @Positive
    void lightweightPaint(Graphics g);

    @Positive
    void paintHeavyweightComponents(Graphics g);

    @Positive
    public void printComponents(Graphics g);

    @Positive
    void lightweightPrint(Graphics g);

    @Positive
    void printHeavyweightComponents(Graphics g);

    @Positive
    public synchronized void addContainerListener(ContainerListener l);

    @Positive
    public synchronized void removeContainerListener(ContainerListener l);

    @Positive
    public synchronized ContainerListener[] getContainerListeners();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected void processContainerEvent(ContainerEvent e);

    @Positive
    void dispatchEventImpl(AWTEvent e);

    @Positive
    void dispatchEventToSelf(AWTEvent e);

    @Positive
    Component getMouseEventTarget(int x, int y, boolean includeSelf);

    @Positive
    Component getDropTargetEventTarget(int x, int y, boolean includeSelf);

    @Positive
    static interface EventTargetFilter {

    @Positive
        boolean accept(final Component comp);
    @Positive
    }

    @Positive
    static class MouseEventTargetFilter implements EventTargetFilter {

    @Positive
        public boolean accept(final Component comp);
    @Positive
    }

    @Positive
    static class DropTargetEventTargetFilter implements EventTargetFilter {

    @Positive
        public boolean accept(final Component comp);
    @Positive
    }

    @Positive
    void proxyEnableEvents(long events);

    @Positive
    @Deprecated
    @Positive
    public void deliverEvent(Event e);

    @Positive
    @Nullable
    @Positive
    public Component getComponentAt(int x, int y);

    @Positive
    @Deprecated
    @Positive
    @Nullable
    @Positive
    public Component locate(int x, int y);

    @Positive
    @Nullable
    @Positive
    public Component getComponentAt(Point p);

    @Positive
    public Point getMousePosition(boolean allowChildren) throws HeadlessException;

    @Positive
    boolean isSameOrAncestorOf(Component comp, boolean allowChildren);

    @Positive
    public Component findComponentAt(int x, int y);

    @Positive
    final Component findComponentAt(int x, int y, boolean ignoreEnabled);

    @Positive
    final Component findComponentAtImpl(int x, int y, boolean ignoreEnabled);

    @Positive
    public Component findComponentAt(Point p);

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    public boolean isAncestorOf(Component c);

    @Positive
    static final class WakingRunnable implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    protected String paramString();

    @Positive
    public void list(PrintStream out, int indent);

    @Positive
    public void list(PrintWriter out, int indent);

    @Positive
    public void setFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    @Positive
    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    @Positive
    public boolean areFocusTraversalKeysSet(int id);

    @Positive
    public boolean isFocusCycleRoot(Container container);

    @Positive
    final boolean containsFocus();

    @Positive
    void clearMostRecentFocusOwnerOnHide();

    @Positive
    void clearCurrentFocusCycleRootOnHide();

    @Positive
    final Container getTraversalRoot();

    @Positive
    public void setFocusTraversalPolicy(FocusTraversalPolicy policy);

    @Positive
    public FocusTraversalPolicy getFocusTraversalPolicy();

    @Positive
    public boolean isFocusTraversalPolicySet();

    @Positive
    public void setFocusCycleRoot(boolean focusCycleRoot);

    @Positive
    public boolean isFocusCycleRoot();

    @Positive
    public final void setFocusTraversalPolicyProvider(boolean provider);

    @Positive
    public final boolean isFocusTraversalPolicyProvider();

    @Positive
    public void transferFocusDownCycle();

    @Positive
    void preProcessKeyEvent(KeyEvent e);

    @Positive
    void postProcessKeyEvent(KeyEvent e);

    @Positive
    boolean postsOldMouseEvents();

    @Positive
    public void applyComponentOrientation(ComponentOrientation o);

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    @Positive
    protected class AccessibleAWTContainer extends AccessibleAWTComponent {

    @Positive
        protected AccessibleAWTContainer() {
    @Positive
        }

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);

    @Positive
        public Accessible getAccessibleAt(Point p);

    @Positive
        @SuppressWarnings("serial")
    @Positive
        protected ContainerListener accessibleContainerHandler;

    @Positive
        protected class AccessibleContainerHandler implements ContainerListener, Serializable {

    @Positive
            protected AccessibleContainerHandler() {
    @Positive
            }

    @Positive
            public void componentAdded(ContainerEvent e);

    @Positive
            public void componentRemoved(ContainerEvent e);
    @Positive
        }

    @Positive
        public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
        public void removePropertyChangeListener(PropertyChangeListener listener);
    @Positive
    }

    @Positive
    Accessible getAccessibleAt(Point p);

    @Positive
    int getAccessibleChildrenCount();

    @Positive
    Accessible getAccessibleChild(int i);

    @Positive
    final void increaseComponentCount(Component c);

    @Positive
    final void decreaseComponentCount(Component c);

    @Positive
    @Override
    @Positive
    final Region getOpaqueShape();

    @Positive
    final void recursiveSubtractAndApplyShape(Region shape);

    @Positive
    final void recursiveSubtractAndApplyShape(Region shape, int fromZorder);

    @Positive
    final void recursiveSubtractAndApplyShape(Region shape, int fromZorder, int toZorder);

    @Positive
    final void recursiveApplyCurrentShape();

    @Positive
    final void recursiveApplyCurrentShape(int fromZorder);

    @Positive
    final void recursiveApplyCurrentShape(int fromZorder, int toZorder);

    @Positive
    final boolean isRecursivelyVisibleUpToHeavyweightContainer();

    @Positive
    @Override
    @Positive
    void mixOnShowing();

    @Positive
    @Override
    @Positive
    void mixOnHiding(boolean isLightweight);

    @Positive
    @Override
    @Positive
    void mixOnReshaping();

    @Positive
    @Override
    @Positive
    void mixOnZOrderChanging(int oldZorder, int newZorder);

    @Positive
    @Override
    @Positive
    void mixOnValidating();
    @Positive
}

    @Positive
class LightweightDispatcher implements java.io.Serializable, AWTEventListener {

    @Positive
    void dispose();

    @Positive
    void enableEvents(long events);

    @Positive
    boolean dispatchEvent(AWTEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void eventDispatched(AWTEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    void retargetMouseEvent(Component target, int id, MouseEvent e);
    @Positive
}

// CFWR semantic augmentation - variant 1

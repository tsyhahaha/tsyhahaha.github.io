const ATTR_DISPLAY = 'sidebar-display';
const ATTR_COLLAPSED = 'sidebar-collapsed';
const STORAGE_KEY = 'sidebar-collapsed';
const $sidebar = document.getElementById('sidebar');
const $mobileTrigger = document.getElementById('sidebar-trigger');
const $desktopTrigger = document.getElementById('sidebar-desktop-trigger');
const $desktopTriggerIcon = $desktopTrigger?.querySelector('.sidebar-toggle-icon');
const $mask = document.getElementById('mask');

class SidebarUtil {
  static #isExpanded = false;
  static #isCollapsed = document.body.hasAttribute(ATTR_COLLAPSED);

  static toggleMobile() {
    this.#isExpanded = !this.#isExpanded;
    document.body.toggleAttribute(ATTR_DISPLAY, this.#isExpanded);
    $sidebar.classList.toggle('z-2', this.#isExpanded);
    $mask.classList.toggle('d-none', !this.#isExpanded);
  }

  static toggleDesktop() {
    this.#isCollapsed = !this.#isCollapsed;
    document.body.toggleAttribute(ATTR_COLLAPSED, this.#isCollapsed);
    this.#syncDesktopTrigger();

    try {
      sessionStorage.setItem(STORAGE_KEY, String(this.#isCollapsed));
    } catch {
      return;
    }
  }

  static init() {
    this.#syncDesktopTrigger();
  }

  static #syncDesktopTrigger() {
    if ($desktopTrigger === null) {
      return;
    }

    const isExpanded = !this.#isCollapsed;
    const actionLabel = isExpanded ? 'Collapse sidebar' : 'Expand sidebar';

    $desktopTrigger.dataset.state = isExpanded ? 'expanded' : 'collapsed';
    $desktopTrigger.setAttribute('aria-expanded', String(isExpanded));
    $desktopTrigger.setAttribute('aria-label', actionLabel);
    $desktopTrigger.setAttribute('title', actionLabel);

    if ($desktopTriggerIcon) {
      $desktopTriggerIcon.classList.toggle('fa-angle-left', isExpanded);
      $desktopTriggerIcon.classList.toggle('fa-angle-right', !isExpanded);
    }
  }
}

export function initSidebar() {
  SidebarUtil.init();

  if ($mobileTrigger !== null && $mask !== null) {
    $mobileTrigger.onclick = $mask.onclick = () => SidebarUtil.toggleMobile();
  }

  if ($desktopTrigger !== null) {
    $desktopTrigger.onclick = () => SidebarUtil.toggleDesktop();
  }
}
